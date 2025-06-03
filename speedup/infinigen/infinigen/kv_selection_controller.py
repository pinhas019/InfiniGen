import torch
import torch.nn.functional as F
from .mlp_predictor import select_kv_mlp


def select_kv(prefetch_idx, k_cache, v_cache):
    """Selects and aggregates critical KV caches using speculated indices

    On the decoding stage, aggregates the critical KV caches corresponding to
    the speculated prefetch index using embedding function.

    Args:
        prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
        k_cache: Key cache (n, bh, d)
        v_cache: Value cache (n, bh, d)

    Returns:
        selected_k: selected key cache (n', bh, d)
        selected_v: selected value cache (n', bh, d)
    """

    prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
    ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1])[None, :]
    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
    return selected_k, selected_v


def speculate_attention(hidden, p_w_q, p_k_c, n_head, alpha, max_num_kv):
    """Speculates the indices of the critical KV caches of next attention layer.

    On the decoding stage, by using the hidden states (layer i), partial query
    weight (layer i+1), and partial key cache (layer i+1), speculates the
    attention score of the next layer. After that, counts the number of
    critical tokens and gets the indcies of the top-k KV cache tokens with high
    attention scores.

    Args:
        hidden: Hidden states of layer i (b, 1, D)
        p_w_q: Partial query weight (D', D)
        p_k_c: Partial key cache (n, bh, d')

        Note that bh * d' == D'

    Returns:
        prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
    """
    b = hidden.shape[0]
    p_q = F.linear(hidden, p_w_q, bias=None)
    p_q = p_q.view(b, 1, n_head, -1)
    p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)

    p_attn = torch.bmm(p_q, p_k_c.permute(1, 2, 0))
    max_ = torch.max(p_attn, dim=-1)[0]
    thr_ = (max_ - alpha).unsqueeze(-1).repeat(1, 1, p_attn.shape[-1])
    count = torch.where(
        p_attn > thr_, torch.ones_like(p_attn), torch.zeros_like(p_attn)
    )
    mean = torch.mean(torch.sum(count, dim=-1)).item()
    prefetch_idx = torch.topk(
        p_attn.permute(2, 1, 0), min(int(mean), max_num_kv), dim=0
    )[1]

    return prefetch_idx


def extract_features_for_prediction(hidden, token_idx, position, layer_idx, attention_scores=None):
    """
    Extract features from hidden states and other information for MLP prediction.
    
    Args:
        hidden: Hidden states tensor
        token_idx: Index of the token
        position: Position in the sequence
        layer_idx: Current layer index
        attention_scores: Optional attention scores
        
    Returns:
        features: Tensor of features for prediction
    """
    # Basic features
    features = [
        float(token_idx),
        float(position),
        float(layer_idx)
    ]
    
    # Add hidden state statistics
    if hidden is not None:
        hidden_flat = hidden.detach().cpu().numpy().flatten()
        features.extend([
            float(np.mean(hidden_flat)),
            float(np.std(hidden_flat)),
            float(np.min(hidden_flat)),
            float(np.max(hidden_flat))
        ])
    
    # Add attention statistics if available
    if attention_scores is not None:
        att = attention_scores.detach().cpu().numpy().flatten()
        features.extend([
            float(np.mean(att)),
            float(np.std(att)),
            float(np.min(att)),
            float(np.max(att))
        ])
    
    # Convert to tensor
    return torch.tensor(features, dtype=torch.float32)
