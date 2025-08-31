import torch

class FeatureCache:
    def __init__(self, num_heads, max_seq_len, device):
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.device = device
        self.recency = torch.zeros((num_heads, max_seq_len), device=self.device)
        self.frequency = torch.zeros((num_heads, max_seq_len), device=self.device)
        self.is_cached = torch.zeros((num_heads, max_seq_len), dtype=torch.bool, device=self.device)

    def update(self, fetch_mask, current_index):
        if current_index == 0:
            self.is_cached[:, 0] = True
            return

        # Update recency for existing tokens
        self.recency[:, :current_index] += 1

        # Update frequency based on the new fetches for the current token
        new_fetches = fetch_mask[:, current_index, :current_index]
        self.frequency[:, :current_index] += new_fetches.float()

        # Add the new token to the cache
        self.is_cached[:, current_index] = True
        self.recency[:, current_index] = 0
        self.frequency[:, current_index] = 1 # Start with a frequency of 1

    def get_features(self, current_index):
        # Consider tokens up to current_index (inclusive)
        cached_mask = self.is_cached[:, : current_index + 1]
    
        # compute raw features
        recency_features = 1.0 / (self.recency[:, : current_index + 1] + 1.0)
        freq_features = self.frequency[:, : current_index + 1].to(recency_features.dtype)
    
        # safe counts (avoid division by zero)
        cached_mask_f = cached_mask.to(recency_features.dtype)
        counts = torch.sum(cached_mask_f, dim=1, keepdim=True).clamp_min(1.0)
    
        # recency normalization (compute mean/std only over cached entries)
        mean_recency = torch.sum(recency_features * cached_mask_f, dim=1, keepdim=True) / counts
        var_recency = torch.sum(((recency_features - mean_recency) * cached_mask_f) ** 2, dim=1, keepdim=True) / counts
        std_recency = torch.sqrt(var_recency)
        norm_recency = (recency_features - mean_recency) / (std_recency + 1e-8)
        # zero out entries for non-cached tokens
        norm_recency = norm_recency * cached_mask_f
    
        # frequency normalization
        mean_freq = torch.sum(freq_features * cached_mask_f, dim=1, keepdim=True) / counts
        var_freq = torch.sum(((freq_features - mean_freq) * cached_mask_f) ** 2, dim=1, keepdim=True) / counts
        std_freq = torch.sqrt(var_freq)
        norm_freq = (freq_features - mean_freq) / (std_freq + 1e-8)
        norm_freq = norm_freq * cached_mask_f
    
        # output shape: (num_heads, seq_len_up_to_current, 2)
        return torch.stack([norm_recency, norm_freq], dim=-1)
