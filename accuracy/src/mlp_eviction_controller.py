import torch
import torch.nn as nn

class EvictionMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1):
        super(EvictionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x.to(self.fc1.weight.dtype))
        out = self.relu(out)
        out = self.fc2(out)
        return out

@torch.no_grad()
def get_eviction_indices(attn, mlp_model, feature_cache, current_index, eviction_batch_size=16):
    """
    Determines which indices to evict based on an MLP model.
    
    Args:
        attn (torch.Tensor): The attention scores tensor.
        mlp_model (nn.Module): The trained MLP for eviction scoring.
        feature_cache (FeatureCache): The cache for recency and frequency features.
        current_index (int): The current sequence index.
        eviction_batch_size (int): The number of tokens to evict at once.

    Returns:
        torch.Tensor: The modified attention tensor with evicted entries set to a large negative number.
    """
    if current_index < 128 or (current_index + 1) % eviction_batch_size != 0:
        return attn

    # Get features from the cache (shape: num_heads x cur_len x 2)
    features = feature_cache.get_features(current_index)

    # ensure k is not larger than available tokens
    cur_len = features.size(1)
    k = min(eviction_batch_size, cur_len)

    # Run MLP on the device expected by the MLP, then bring scores back to attn.device
    mlp_device = None
    try:
        mlp_device = next(mlp_model.parameters()).device
    except StopIteration:
        mlp_device = attn.device

    features_for_mlp = features.to(mlp_device)
    scores = mlp_model(features_for_mlp)
    # scores: (num_heads, cur_len, 1) -> squeeze to (num_heads, cur_len)
    scores = scores.squeeze(-1).to(attn.device)

    # Evict tokens with the lowest scores (per-head)
    # topk requires k <= cur_len
    _, evict_indices = torch.topk(scores, k=k, dim=1, largest=False)
    evict_indices = evict_indices.long()

    # If there are no future target positions, nothing to do
    H, T, S = attn.shape
    start_t = current_index + 1
    if start_t >= T:
        return attn

    t_after = torch.arange(start_t, T, device=attn.device)

    # Build broadcasted index tensors of shape (H, T_after, k)
    H_idx = torch.arange(H, device=attn.device).view(H, 1, 1).expand(H, t_after.numel(), k)
    T_idx = t_after.view(1, t_after.numel(), 1).expand(H, t_after.numel(), k)
    S_idx = evict_indices.unsqueeze(1).expand(H, t_after.numel(), k)

    # Apply eviction (set very negative attention for chosen source indices)
    attn[H_idx, T_idx, S_idx] = -10000.0

    return attn
