"""
Usage:
python3 -m flexgen.flex_opt --model facebook/opt-1.3b --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""

import argparse
import dataclasses
import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from flexgen.compression import CompressionConfig
from flexgen.opt_config import OptConfig, get_opt_config, download_opt_weights
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import, TorchTensor)
from flexgen.timer import timers
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

from infinigen.skewing_controller import weight_bias_concat
from infinigen.kv_selection_controller import select_kv, select_kv_mlp, extract_features_for_prediction
from infinigen.partial_weight_generation_controller import set_partial_cache, set_partial_weight
from infinigen.mlp_predictor import KVBlockPredictor
from infinigen.mlp_integration import initialize_mlp_predictor

fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret


class InputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
            # w_pos
            ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, w_pos = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.compute.opt_input_embed(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate)
        hidden.val = h


class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "decoder.layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "decoder.layer_norm.bias"),
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, b_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 4
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()
        else:
            (w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val

        h = self.compute.opt_output_embed(h, w_ln, b_ln, w_token, donate,
            self.task.do_sample, self.task.temperature)
        hidden.val = h


class SelfAttention:
    def __init__(self, config, env, policy, layer_id, enable_prefetching, partial_weight_ratio=0.2, alpha=4, max_num_kv=400):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.task = None
        self.enable_prefetching = enable_prefetching
        self.prefetch_idx = None
        self.prefetch_kv = None
        self.partial_index = None
        self.alpha = alpha
        self.max_num_kv = max_num_kv
        if self.layer_id > 1:
            self.partial_weight_ratio = partial_weight_ratio
        else:
            self.partial_weight_ratio = None
            
        # Initialize MLP predictor
        self.use_mlp_predictor = True  # Enable MLP predictor by default
        self.mlp_predictor = None
        self.last_hidden = None

    def set_task(self, task):
        self.task = task
        
        # Initialize MLP predictor when task is set
        if self.use_mlp_predictor and self.mlp_predictor is None:
            # Define input dimension based on feature extraction
            # Basic features (token_idx, position, layer_idx) + 
            # hidden state stats (mean, std, min, max) +
            # attention stats (mean, std, min, max)
            input_dim = 3 + 4 + 4
            
            # Hidden dimension can be adjusted based on model size
            hidden_dim = 128
            
            # Output dimension is 1 (probability of being needed)
            output_dim = 1
            
            # Create predictor
            self.mlp_predictor = KVBlockPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                device=self.compute.dev,
                max_cache_size=self.max_num_kv
            )

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out
            ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        # WQ
        head_dim = h // self.config.n_head
        weights[0].data = weight_bias_concat(weights[0].data, weights[1].data, True, head_dim)
        weights[0].shape = (h, h+1)
        # WK
        weights[2].data = weight_bias_concat(weights[2].data, weights[3].data)
        weights[2].shape = (h, h+1)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)
        if self.layer_id > 1:
            self.prefetch_kv = device.allocate((2, self.max_num_kv, cache_home.val[0].shape[1], cache_home.val[0].shape[2]), np.float16, pin_memory=True)

    def load_cache(self, cache_home, cache_read_buf, i):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val
        
        # Use MLP predictor for token selection if enabled and in generation phase
        if self.use_mlp_predictor and i > 1 and self.mlp_predictor is not None and self.last_hidden is not None:
            # Determine how many tokens to select
            num_tokens = min(self.max_num_kv, self.task.prompt_len + i - 1)
            
            # Use predictor to select tokens
            selected_indices = []
            for idx in range(self.task.prompt_len + i - 1):
                # Extract features for this token
                features = extract_features_for_prediction(
                    self.last_hidden,
                    idx,
                    idx,  # position = idx
                    self.layer_id
                )
                
                # Predict importance
                with torch.no_grad():
                    importance = self.mlp_predictor.predict(features.to(self.compute.dev))
                    
                if importance.item() > 0.5:  # Threshold can be adjusted
                    selected_indices.append(idx)
                    
                # Limit number of selected tokens
                if len(selected_indices) >= num_tokens:
                    break
            
            # If tokens were selected, use them
            if selected_indices:
                dst = self.attention_compute
                selected_k, selected_v = select_kv_mlp(
                    self.mlp_predictor,
                    k_home,
                    v_home,
                    selected_indices
                )
                
                if selected_k is not None and selected_v is not None:
                    cache_read_buf.store((
                        (selected_k, False),
                        (selected_v, False),
                    ))
                    return
        
        # Fall back to original implementation if MLP predictor is not enabled
        # or if no tokens were selected

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, self.task.prompt_len + i),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")
    
    def prefetch_cache(self, cache_home, cache_read_buf, i, prefetch_idx, prefetch_cache_stream):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, prefetch_idx.shape[0]),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                self.prefetch_kv.data[0, :prefetch_idx.shape[0]], self.prefetch_kv.data[1, :prefetch_idx.shape[0]] = select_kv(prefetch_idx, k_home.data, v_home.data)
                k_c = TorchTensor((prefetch_idx.shape[0], k_home.shape[1], k_home.shape[2]), k_home.dtype, self.prefetch_kv.data[0, :prefetch_idx.shape[0]], k_home.device)
                v_c = TorchTensor((prefetch_idx.shape[0], v_home.shape[1], v_home.shape[2]), v_home.dtype, self.prefetch_kv.data[1, :prefetch_idx.shape[0]], v_home.device)

                with torch.cuda.stream(prefetch_cache_stream):
                    cache_read_buf.store((
                        k_c.smart_copy(dst, indices),
                        v_c.smart_copy(dst, indices),
                    ))
        elif path == 1 or path == 2:
            raise ValueError(f"Not implemented path: {path}")
        else:    
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)
        
        # Update MLP predictor with new token if in generation phase
        if i > 0 and self.use_mlp_predictor and self.mlp_predictor is not None and self.last_hidden is not None:
            # Current token index
            current_idx = self.task.prompt_len + i - 1
            
            # Extract features
            features = extract_features_for_prediction(
                self.last_hidden,
                current_idx,
                current_idx,  # position = idx
                self.layer_id
            )
            
            # Get k and v values for this token
            if hasattr(k_new.data, '__getitem__') and hasattr(v_new.data, '__getitem__'):
                k_value = k_new.data[0].clone() if k_new.shape[0] > 0 else None
                v_value = v_new.data[0].clone() if v_new.shape[0] > 0 else None
                
                if k_value is not None and v_value is not None:
                    # Update predictor
                    self.mlp_predictor.update(
                        current_idx,
                        k_value,
                        v_value,
                        features.to(self.compute.dev)
                    )
                    
                    # Train predictor periodically
                    if i % 10 == 0:  # Adjust frequency as needed
                        self.mlp_predictor.train(num_batches=1)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, warmup, partial_weight_read_buf, partial_cache_read_buf, speculation_stream, prev_partial_cache_read_buf, prev_partial_weight_read_buf, weight_home):
        # Store hidden for feature extraction
        self.last_hidden = hidden.val
        
        n_head = self.config.n_head

        donate = [False] * 14
        h, donate[0] = hidden.val, True
        head_dim = h.shape[-1] // n_head

        if i == 0:  # prefill
            # Load attention mask
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)

            # Load weights
            if k == self.policy.num_gpu_batches - 1:
                # Clear the weight_read_buf if it is the last gpu batch
                (w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]), (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]), (w_ln, donate[10]), (b_ln, donate[11]) = weight_read_buf.pop()
            else:
                (w_q, _), (b_q, _), (w_k, _), (b_k, _), (w_v, _), (b_v, _), (w_out, _), (b_out, _), (w_ln, _), (b_ln, _) = weight_read_buf.val

            # Compute attention
            h, k_cache, v_cache, w_q, w_k, partial_index = self.compute.mha(
                h, mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln,
                n_head, donate, self.policy.compress_cache, self.policy.comp_cache_config, warmup, self.partial_weight_ratio)
            self.partial_index = partial_index
            cache_write_buf.store((k_cache, v_cache))
            hidden.val = h
        else:  # decoding
            # Load KV cache
            if self.enable_prefetching and self.layer_id > 1 and prev_partial_cache_read_buf is not None and prev_partial_weight_read_buf is not None:
                p_w_q, p_k_c = prev_partial_weight_read_buf.val
                self.prefetch_idx = self.compute.speculate_attention(h, p_w_q, p_k_c, n_head, self.alpha, self.max_num_kv)
                self.prefetch_cache(cache_home, cache_read_buf, i, self.prefetch_idx, speculation_stream)
            else:
                self.load_cache(cache_home, cache_read_buf, i)

            # Load weights
            if k == self.policy.num_gpu_batches - 1:
                # Clear the weight_read_buf if it is the last gpu batch
                (w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]), (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]), (w_ln, donate[10]), (b_ln, donate[11]) = weight_read_buf.pop()
            else:
                (w_q, _), (b_q, _), (w_k, _), (b_k, _), (w_v, _), (b_v, _), (w_out, _), (b_out, _), (w_ln, _), (b_ln, _) = weight_read_buf.val

            # Load attention mask
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)

            # Load KV cache
            if self.enable_prefetching and self.layer_id > 1 and self.prefetch_idx is not None:
                (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            else:
                (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()

            # Set partial weight
            p_w_q = None
            if self.layer_id > 1 and self.partial_index is not None:
                p_w_q = set_partial_weight(w_q.data, self.partial_index)

            # Set partial cache
            p_k_c = None
            if self.layer_id > 1 and self.partial_index is not None:
                p_k_c = set_partial_cache(k_cache, self.partial_index)

            # Store partial weight and cache
            if self.layer_id > 1 and p_w_q is not None and p_k_c is not None:
                partial_weight_read_buf.store((p_w_q, p_k_c))

            # Compute attention
            h, k_cache_new, v_cache_new = self.compute.mha_gen(
                h, mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln,
                n_head, k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config, p_w_q, p_k_c, speculation_stream, self.alpha, self.max_num_kv)
            cache_write_buf.store((k_cache_new, v_cache_new))
            hidden.val = h
            
    def prefetch_kv_blocks(self, current_tokens, k_cache, v_cache):
        """
        Prefetch KV blocks that are predicted to be needed soon.
        
        Args:
            current_tokens: List of current token indices
            k_cache: Key cache
            v_cache: Value cache
            
        Returns:
            prefetched: List of prefetched token indices
        """
        if not self.use_mlp_predictor or self.mlp_predictor is None or self.last_hidden is None:
            return []
        
        # Extract features for all tokens in context
        features_list = []
        for idx in range(len(current_tokens) + 100):  # Look ahead 100 tokens
            if idx >= k_cache.shape[0]:
                break
                
            features = extract_features_for_prediction(
                self.last_hidden,
                idx,
                idx,  # position = idx
                self.layer_id
            )
            features_list.append(features.to(self.compute.dev))
        
        # Prefetch tokens
        return self.mlp_predictor.prefetch(
            current_tokens,
            features_list,
            k_cache,
            v_cache
        )


class MLP:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.mlp"))
        weight_specs = [
            # w_in
            ((h, 4 * h), dtype, path + ".fc1.weight"),
            # b_in
            ((4 * h,), dtype, path + ".fc1.bias"),
            # w_out
            ((4 * h, h), dtype, path + ".fc2.weight"),
            # b_out
            ((h,), dtype, path + ".fc2.bias"),
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_in, b_in, w_out, b_out, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_in.smart_copy(dst1), b_in.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_in, donate[1]), (b_in, donate[2]), (w_out, donate[3]), (b_out, donate[4]), (w_ln, donate[5]), (b_ln, donate[6]) = weight_read_buf.pop()
        else:
            (w_in, _), (b_in, _), (w_out, _), (b_out, _), (w_ln, _), (b_ln, _) = weight_read_buf.val

        h = self.compute.opt_mlp(h, w_in, b_in, w_out, b_out, w_ln, b_ln, donate)
        hidden.val = h


class TransformerLayer:
    def __init__(self, config, env, policy, layer_id, enable_prefetching, partial_weight_ratio=0.2, alpha=4, max_num_kv=400):
        self.policy = policy
        self.compute = env.gpu

        if policy.sep_layer:
            self.attn = SelfAttention(config, env, policy, layer_id, enable_prefetching, partial_weight_ratio, alpha, max_num_kv)
            self.mlp = MLP(config, env, policy, layer_id)
            self.sub_layers = [self.attn, self.mlp]
        else:
            raise NotImplementedError("Not implemented yet")

    def set_task(self, task):
        for sub_layer in self.sub_layers:
            sub_layer.set_task(task)

    def init_weight(self, weight_home, path):
        sub_weight_homes = array_1d(len(self.sub_layers))
        for i, sub_layer in enumerate(self.sub_layers):
            sub_layer.init_weight(sub_weight_homes[i], path)
        weight_home.store(sub_weight_homes)

    def load_weight(self, weight_home, weight_read_buf, k):
        sub_weight_homes = weight_home.val
        sub_weight_read_bufs = array_1d(len(self.sub_layers))
        for i, sub_layer in enumerate(self.sub_layers):
            sub_layer.load_weight(sub_weight_homes[i], sub_weight_read_bufs[i], k)
        weight_read_buf.store(sub_weight_read_bufs)

    def init_cache_one_gpu_batch(self, cache_home):
        sub_cache_homes = array_1d(len(self.sub_layers))
        for i, sub_layer in enumerate(self.sub_layers):
            sub_layer.init_cache_one_gpu_batch(sub_cache_homes[i])
        cache_home.store(sub_cache_homes)

    def load_cache(self, cache_home, cache_read_buf, i):
        sub_cache_homes = cache_home.val
        sub_cache_read_bufs = array_1d(len(self.sub_layers))
        for i, (sub_layer, sub_cache_home, sub_cache_read_buf) in enumerate(
                zip(self.sub_layers, sub_cache_homes, sub_cache_read_bufs)):
            sub_layer.load_cache(sub_cache_home, sub_cache_read_buf, i)
        cache_read_buf.store(sub_cache_read_bufs)

    def store_cache(self, cache_home, cache_write_buf, i):
        sub_cache_homes = cache_home.val
        sub_cache_write_bufs = cache_write_buf.val
        for sub_layer, sub_cache_home, sub_cache_write_buf in zip(
                self.sub_layers, sub_cache_homes, sub_cache_write_bufs):
            sub_layer.store_cache(sub_cache_home, sub_cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, warmup, partial_weight_read_buf, partial_cache_read_buf, speculation_stream, prev_partial_cache_read_buf, prev_partial_weight_read_buf, weight_home):
        # Unpack read buffers
        sub_cache_read_bufs = cache_read_buf.val
        sub_weight_read_bufs = weight_read_buf.val

        # Unpack write buffers
        sub_cache_write_bufs = array_1d(len(self.sub_layers))
        cache_write_buf.store(sub_cache_write_bufs)

        # Forward for attention
        self.attn.forward(hidden, sub_cache_read_bufs[0], sub_weight_read_bufs[0],
            attention_mask, sub_cache_write_bufs[0], i, k, warmup, partial_weight_read_buf, partial_cache_read_buf, speculation_stream, prev_partial_cache_read_buf, prev_partial_weight_read_buf, weight_home)

        # Forward for MLP
        self.mlp.forward(hidden, sub_cache_read_bufs[1], sub_weight_read_bufs[1],
            attention_mask, sub_cache_write_bufs[1], i, k)


class Transformer:
    def __init__(self, config, env, policy, enable_prefetching, partial_weight_ratio=0.2, alpha=4, max_num_kv=400):
        self.input_embed = InputEmbed(config, env, policy)
        self.layers = []
        for i in range(config.num_hidden_layers):
            self.layers.append(TransformerLayer(config, env, policy, i, enable_prefetching, partial_weight_ratio, alpha, max_num_kv))
        self.output_embed = OutputEmbed(config, env, policy)
        self.policy = policy

    def set_task(self, task):
        self.input_embed.set_task(task)
        for layer in self.layers:
            layer.set_task(task)
        self.output_embed.set_task(task)

    def init_weight(self, weight_home, path):
        sub_weight_homes = array_1d(len(self.layers) + 2)
        self.input_embed.init_weight(sub_weight_homes[0], path)
        for i, layer in enumerate(self.layers):
            layer.init_weight(sub_weight_homes[i + 1], path)
        self.output_embed.init_weight(sub_weight_homes[-1], path)
        weight_home.store(sub_weight_homes)

    def load_weight(self, weight_home, weight_read_buf, k):
        sub_weight_homes = weight_home.val
        sub_weight_read_bufs = array_1d(len(self.layers) + 2)
        self.input_embed.load_weight(sub_weight_homes[0], sub_weight_read_bufs[0], k)
        for i, layer in enumerate(self.layers):
            layer.load_weight(sub_weight_homes[i + 1], sub_weight_read_bufs[i + 1], k)
        self.output_embed.load_weight(sub_weight_homes[-1], sub_weight_read_bufs[-1], k)
        weight_read_buf.store(sub_weight_read_bufs)

    def init_cache_one_gpu_batch(self, cache_home):
        sub_cache_homes = array_1d(len(self.layers) + 2)
        self.input_embed.init_cache_one_gpu_batch(sub_cache_homes[0])
        for i, layer in enumerate(self.layers):
            layer.init_cache_one_gpu_batch(sub_cache_homes[i + 1])
        self.output_embed.init_cache_one_gpu_batch(sub_cache_homes[-1])
        cache_home.store(sub_cache_homes)

    def load_cache(self, cache_home, cache_read_buf, i):
        sub_cache_homes = cache_home.val
        sub_cache_read_bufs = array_1d(len(self.layers) + 2)
        self.input_embed.load_cache(sub_cache_homes[0], sub_cache_read_bufs[0], i)
        for i, (layer, sub_cache_home, sub_cache_read_buf) in enumerate(
                zip(self.layers, sub_cache_homes[1:-1], sub_cache_read_bufs[1:-1])):
            layer.load_cache(sub_cache_home, sub_cache_read_buf, i)
        self.output_embed.load_cache(sub_cache_homes[-1], sub_cache_read_bufs[-1], i)
        cache_read_buf.store(sub_cache_read_bufs)

    def store_cache(self, cache_home, cache_write_buf, i):
        sub_cache_homes = cache_home.val
        sub_cache_write_bufs = cache_write_buf.val
        self.input_embed.store_cache(sub_cache_homes[0], sub_cache_write_bufs[0], i)
        for i, (layer, sub_cache_home, sub_cache_write_buf) in enumerate(
                zip(self.layers, sub_cache_homes[1:-1], sub_cache_write_bufs[1:-1])):
            layer.store_cache(sub_cache_home, sub_cache_write_buf, i)
        self.output_embed.store_cache(sub_cache_homes[-1], sub_cache_write_bufs[-1], i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, warmup=False):
        # Unpack read buffers
        sub_cache_read_bufs = cache_read_buf.val
        sub_weight_read_bufs = weight_read_buf.val

        # Unpack write buffers
        sub_cache_write_bufs = array_1d(len(self.layers) + 2)
        cache_write_buf.store(sub_cache_write_bufs)

        # Forward for input embedding
        self.input_embed.forward(hidden, sub_cache_read_bufs[0], sub_weight_read_bufs[0],
            attention_mask, sub_cache_write_bufs[0], i, k)

        # Forward for layers
        partial_weight_read_bufs = array_1d(len(self.layers))
        partial_cache_read_bufs = array_1d(len(self.layers))
        speculation_stream = torch.cuda.Stream()
        for j, layer in enumerate(self.layers):
            prev_partial_cache_read_buf = None
            prev_partial_weight_read_buf = None
            if j > 0:
                prev_partial_cache_read_buf = partial_cache_read_bufs[j-1]
                prev_partial_weight_read_buf = partial_weight_read_bufs[j-1]
            layer.forward(hidden, sub_cache_read_bufs[j + 1], sub_weight_read_bufs[j + 1],
                attention_mask, sub_cache_write_bufs[j + 1], i, k, warmup, partial_weight_read_bufs[j], partial_cache_read_bufs[j], speculation_stream, prev_partial_cache_read_buf, prev_partial_weight_read_buf, sub_weight_read_bufs[j + 1])

        # Forward for output embedding
        self.output_embed.forward(hidden, sub_cache_read_bufs[-1], sub_weight_read_bufs[-1],
            attention_mask, sub_cache_write_bufs[-1], i, k)


class FlexOPTInfinigen:
    def __init__(self, model_name, env, warmup_batch_size=1, partial_weight_ratio=0.2, alpha=4, max_num_kv=400):
        self.model_name = model_name
        self.env = env
        self.warmup_batch_size = warmup_batch_size
        self.partial_weight_ratio = partial_weight_ratio
        self.alpha = alpha
        self.max_num_kv = max_num_kv

        # Download weights
        self.config = get_opt_config(model_name)
        self.path = download_opt_weights(model_name, self.config)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_policy(self, task, gpu_batch_size, cpu_cache_compute=False):
        assert task.prompt_len + task.gen_len <= self.config.max_seq_len

        # Compute the maximum number of GPU batches
        max_num_gpu_batches = task.batch_size // gpu_batch_size
        if task.batch_size % gpu_batch_size != 0:
            max_num_gpu_batches += 1

        # Compute the cache size
        if task.gen_len == 0:  # prefill only
            max_gpu_cache_size = (self.config.n_head *
                self.config.input_dim // self.config.n_head *
                task.prompt_len * gpu_batch_size * 2 * 2)
        else:  # prefill + decoding
            max_gpu_cache_size = (self.config.n_head *
                self.config.input_dim // self.config.n_head *
                (task.prompt_len + task.gen_len) * gpu_batch_size * 2 * 2)

        # Compute the weight size
        max_gpu_weight_size = 0
        for i in range(self.config.num_hidden_layers):
            max_gpu_weight_size += (self.config.input_dim * self.config.input_dim * 4 + # attn
                                   self.config.input_dim * self.config.input_dim * 4 + # mlp
                                   self.config.input_dim * 12) * 2  # others

        # Compute the workspace size
        # FIXME(woosuk): This is an empirical value. Need to analyze the accurate memory usage.
        max_gpu_workspace_size = (self.config.input_dim * task.batch_size *
            (task.prompt_len + task.gen_len) * 4 * 8)

        # Compute the available memory for weights and cache
        gpu_available = self.env.gpu.mem_capacity
        if gpu_available <= max_gpu_workspace_size:
            raise ValueError("GPU memory is not enough for the workspace.")
        gpu_available -= max_gpu_workspace_size

        # Compute the weight and cache percentages
        if gpu_available >= max_gpu_weight_size + max_gpu_cache_size:
            # Enough GPU memory
            weight_gpu_percent = 100
            cache_gpu_percent = 100
        elif gpu_available >= max_gpu_weight_size:
            # Enough GPU memory for weights
            weight_gpu_percent = 100
            cache_gpu_percent = int(100 * (gpu_available - max_gpu_weight_size) / max_gpu_cache_size)
        else:
            # Not enough GPU memory for weights
            weight_gpu_percent = int(100 * gpu_available / max_gpu_weight_size)
            cache_gpu_percent = 0

        # Create a policy
        # FIXME(woosuk): Support disk offloading.
        policy = Policy(gpu_batch_size=gpu_batch_size,
                        num_gpu_batches=max_num_gpu_batches,
                        w_gpu_percent=weight_gpu_percent,
                        w_cpu_percent=100 - weight_gpu_percent,
                        cache_gpu_percent=cache_gpu_percent,
                        cache_cpu_percent=100 - cache_gpu_percent,
                        act_gpu_percent=100,
                        act_cpu_percent=0,
                        overlap=True,
                        sep_layer=True,
                        pin_weight=True,
                        cpu_cache_compute=cpu_cache_compute,
                        attn_sparsity=1.0,
                        compress_weight=False,
                        comp_weight_config=CompressionConfig(
                            num_bits=4,
                            group_size=64,
                            group_dim=0,
                            symmetric=False,
                        ),
                        compress_cache=False,
                        comp_cache_config=CompressionConfig(
                            num_bits=4,
                            group_size=64,
                            group_dim=2,
                            symmetric=False,
                        ))
        return policy

    def init_model(self, policy, enable_prefetching=True):
        # Initialize the model
        self.model = Transformer(self.config, self.env, policy, enable_prefetching, self.partial_weight_ratio, self.alpha, self.max_num_kv)

        # Prepare weights
        self.weight_home = ValueHolder()
        self.model.init_weight(self.weight_home, self.path)

    def warmup(self, task, policy):
        # Create dummy inputs
        bs = self.warmup_batch_size
        input_ids = torch.zeros((bs, task.prompt_len), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Create dummy outputs
        hidden_shape, hidden_dtype = self.model.input_embed.input_act_shape_and_dtype(bs, task.prompt_len)
        hidden = self.env.gpu.allocate(hidden_shape, torch_dtype_to_np_dtype[hidden_dtype])
        hidden.data.copy_(input_ids)

        # Create cache
        cache_home = ValueHolder()
        self.model.init_cache_one_gpu_batch(cache_home)

        # Create buffers
        weight_read_buf = ValueHolder()
        cache_read_buf = ValueHolder()
        cache_write_buf = ValueHolder()
        attention_mask = ValueHolder()
        attention_mask.val = self.env.gpu.allocate(
            (bs, task.prompt_len), np.bool_)
        attention_mask.val.data.copy_(torch.ones_like(input_ids))

        # Load weights
        self.model.load_weight(self.weight_home, weight_read_buf, 0)

        # Run the model
        self.model.forward(hidden, cache_read_buf, weight_read_buf, attention_mask,
                          cache_write_buf, 0, 0, True)

    def generate(self, task, policy, enable_prefetching=True):
        # Initialize the model
        self.init_model(policy, enable_prefetching)
        self.model.set_task(task)

        # Warmup
        if task.prompt_len > 1:
            self.warmup(task, policy)

        # Create inputs
        tokenized = self.tokenizer(task.prompts,
                                  padding="max_length",
                                  truncation=True,
                                  max_length=task.prompt_len,
                                  return_tensors="pt")
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # Create outputs
        output_ids = torch.zeros((task.batch_size, task.prompt_len + task.gen_len),
                                dtype=torch.long)
        output_ids[:, :task.prompt_len] = input_ids

        # Create cache
        cache_home = ValueHolder()
        self.model.init_cache_one_gpu_batch(cache_home)

        # Run prefill
        if task.prompt_len > 1:
            # Create buffers
            hidden_shape, hidden_dtype = self.model.input_embed.input_act_shape_and_dtype(
                policy.gpu_batch_size, task.prompt_len)
            hidden = self.env.gpu.allocate(hidden_shape, torch_dtype_to_np_dtype[hidden_dtype])
            weight_read_buf = ValueHolder()
            cache_read_buf = ValueHolder()
            cache_write_buf = ValueHolder()
            attention_mask_buf = ValueHolder()

            # Load weights
            self.model.load_weight(self.weight_home, weight_read_buf, 0)

            # Iterate over batches
            for i in range(0, task.batch_size, policy.gpu_batch_size):
                # The actual batch size for this iteration
                cur_bs = min(policy.gpu_batch_size, task.batch_size - i)

                # Set inputs
                hidden.data.copy_(input_ids[i:i+cur_bs])
                attention_mask_buf.val = self.env.gpu.allocate(
                    (cur_bs, task.prompt_len), np.bool_)
                attention_mask_buf.val.data.copy_(attention_mask[i:i+cur_bs])

                # Load cache
                self.model.load_cache(cache_home, cache_read_buf, 0)

                # Run the model
                self.model.forward(hidden, cache_read_buf, weight_read_buf,
                                  attention_mask_buf, cache_write_buf, 0, 0)

                # Store cache
                self.model.store_cache(cache_home, cache_write_buf, 0)

        # Run decoding
        if task.gen_len > 0:
            # Create buffers
            hidden_shape, hidden_dtype = self.model.input_embed.input_act_shape_and_dtype(
                policy.gpu_batch_size, 1)
            hidden = self.env.gpu.allocate(hidden_shape, torch_dtype_to_np_dtype[hidden_dtype])
            weight_read_buf = ValueHolder()
            cache_read_buf = ValueHolder()
            cache_write_buf = ValueHolder()
            attention_mask_buf = ValueHolder()

            # Load weights
            self.model.load_weight(self.weight_home, weight_read_buf, 0)

            # Iterate over batches
            for i in range(0, task.batch_size, policy.gpu_batch_size):
                # The actual batch size for this iteration
                cur_bs = min(policy.gpu_batch_size, task.batch_size - i)

                # Iterate over generation steps
                for j in range(task.gen_len):
                    # Set inputs
                    pos = task.prompt_len + j
                    hidden.data.copy_(output_ids[i:i+cur_bs, pos-1:pos])
                    attention_mask_buf.val = self.env.gpu.allocate(
                        (cur_bs, pos), np.bool_)
                    attention_mask_buf.val.data.copy_(
                        torch.ones((cur_bs, pos), dtype=torch.bool))

                    # Load cache
                    self.model.load_cache(cache_home, cache_read_buf, j + 1)

                    # Run the model
                    self.model.forward(hidden, cache_read_buf, weight_read_buf,
                                      attention_mask_buf, cache_write_buf, j + 1, 0)

                    # Store cache
                    self.model.store_cache(cache_home, cache_write_buf, j + 1)

                    # Update output
                    output_ids[i:i+cur_bs, pos:pos+1] = hidden.data

        # Decode output
        outputs = []
        for i in range(task.batch_size):
            output = output_ids[i, :task.prompt_len + task.gen_len].tolist()
            outputs.append(self.tokenizer.decode(output))
        return outputs


def run_flexopt_infinigen(args):
    # Create execution env
    env = ExecutionEnv.create(args.cpu_memory_limit, args.gpu_memory_limit)

    # Create task
    task = Task(
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        prompts=args.prompts,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )

    # Create model
    model = FlexOPTInfinigen(args.model, env, args.warmup_batch_size, args.partial_weight_ratio, args.alpha, args.max_num_kv)

    # Create policy
    policy = model.get_policy(task, args.gpu_batch_size, args.cpu_cache_compute)

    # Run generation
    t1 = time.time()
    outputs = model.generate(task, policy, args.enable_prefetching)
    t2 = time.time()
    latency = t2 - t1

    # Print outputs
    for i, output in enumerate(outputs):
        print(f"Output {i}: {output}")
    print(f"Latency: {latency:.2f} s")

    # Print policy
    print(f"Policy: {policy}")

    # Print memory usage
    if args.gpu_memory_limit > 0:
        print(f"GPU memory usage: {torch_mem_stats()}")
    if args.cpu_memory_limit > 0:
        print(f"CPU memory usage: {cpu_mem_stats()}")

    # Benchmark
    if args.bench_decode_step_latency:
        # Benchmark the latency of a single decoding step
        # This is useful for estimating the end-to-end latency
        # for longer sequences
        latency = project_decode_latency(
            model, task, policy, args.enable_prefetching)
        print(f"Projected latency for {args.bench_decode_step_latency} tokens: "
              f"{latency:.2f} s")

    # Write benchmark log
    if args.log_file:
        write_benchmark_log(args.log_file, args, latency)


def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
        help="The model name.")
    parser.add_argument("--warmup-batch-size", type=int, default=1,
        help="The batch size for warmup.")
    parser.add_argument("--partial-weight-ratio", type=float, default=0.2,
        help="The ratio of partial weight.")
    parser.add_argument("--alpha", type=float, default=4,
        help="The alpha value for attention.")
    parser.add_argument("--max-num-kv", type=int, default=400,
        help="The maximum number of KV cache tokens.")
    parser.add_argument("--enable-prefetching", type=str2bool, default=True,
        help="Whether to enable prefetching.")
    parser.add_argument("--cpu-cache-compute", type=str2bool, default=False,
        help="Whether to compute attention on CPU.")
    parser.add_argument("--batch-size", type=int, default=1,
        help="The batch size.")
    parser.add_argument("--gpu-batch-size", type=int, default=0,
        help="The GPU batch size. If 0, use batch-size.")
    parser.add_argument("--prompt-len", type=int, default=512,
        help="The prompt length.")
    parser.add_argument("--gen-len", type=int, default=32,
        help="The generation length.")
    parser.add_argument("--do-sample", type=str2bool, default=False,
        help="Whether to use sampling.")
    parser.add_argument("--temperature", type=float, default=1.0,
        help="The temperature for sampling.")
    parser.add_argument("--cpu-memory-limit", type=int, default=0,
        help="The CPU memory limit in bytes. If 0, no limit.")
    parser.add_argument("--gpu-memory-limit", type=int, default=0,
        help="The GPU memory limit in bytes. If 0, no limit.")
    parser.add_argument("--bench-decode-step-latency", type=int, default=0,
        help="Benchmark the latency of a single decoding step and project the "
             "latency for this many tokens. If 0, no benchmark.")
    parser.add_argument("--log-file", type=str, default="",
        help="The log file to write benchmark results. If empty, no log.")
    parser.add_argument("--prompts", type=str, nargs="+", default=["Hello, I'm a language model"],
        help="The prompts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    if args.gpu_batch_size == 0:
        args.gpu_batch_size = args.batch_size

    run_flexopt_infinigen(args)
