import torch
import torch.nn as nn

class EvictionMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1):
        super(EvictionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def get_eviction_indices(attn, fetch_mask, mlp_model, current_index):
    """
    Determines which indices to evict based on an MLP model.
    
    Args:
        attn (torch.Tensor): The attention scores tensor.
        fetch_mask (torch.Tensor): The mask indicating which KV pairs have been fetched.
        mlp_model (nn.Module): The trained MLP for eviction scoring.
        current_index (int): The current sequence index.

    Returns:
        torch.Tensor: The modified attention tensor with evicted entries set to a large negative number.
    """
    if current_index < 128: # Start evicting only after a certain number of tokens are cached
        return attn

    # Features: recency (inverse of age) and frequency
    cached_indices = torch.arange(current_index + 1, device=attn.device)
    recency = 1.0 / (current_index + 1 - cached_indices)
    frequency = torch.sum(fetch_mask[:, :current_index + 1, :current_index+1], dim=1).float()
    
    # Normalize features
    recency = (recency - recency.mean()) / (recency.std() + 1e-8)
    frequency = (frequency - frequency.mean(dim=-1, keepdim=True)) / (frequency.std(dim=-1, keepdim=True) + 1e-8)

    # Prepare input for MLP
    features = torch.stack([recency.expand_as(frequency), frequency], dim=-1)
    
    # Get scores from MLP
    scores = mlp_model(features)
    
    # Evict token with the lowest score
    _, evict_idx = torch.min(scores, dim=1)
    
    # Apply eviction
    for head_idx in range(attn.shape[0]):
        attn[head_idx, (current_index + 1):, evict_idx[head_idx]] = -10000
    
    return attn
