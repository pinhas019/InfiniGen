import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class KVBlockPredictor:
    """
    MLP-based predictor for KV cache blocks that will be needed in future generation steps.
    
    This predictor uses a simple MLP to learn patterns in token usage and predict
    which KV blocks will be needed in the next generation step, enabling more
    intelligent eviction and prefetching decisions.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, device, 
                 max_cache_size, learning_rate=0.001, buffer_size=10000, 
                 batch_size=64, gamma=0.99):
        """
        Initialize the KV Block Predictor.
        
        Args:
            input_dim: Dimension of input features (e.g., token embeddings, position, etc.)
            hidden_dim: Dimension of hidden layer in the MLP
            output_dim: Dimension of output (typically number of possible KV blocks)
            device: Torch device for tensor operations
            max_cache_size: Maximum number of KV blocks to keep in cache
            learning_rate: Learning rate for optimizer
            buffer_size: Size of experience replay buffer
            batch_size: Batch size for training
            gamma: Discount factor for future rewards
        """
        self.device = device
        self.max_cache_size = max_cache_size
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Create MLP model
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output probability of each block being needed
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Current cache state (token_idx -> (k_value, v_value))
        self.cache = {}
        
        # Access history for each token
        self.access_history = {}
        
        # Feature history for each token
        self.feature_history = {}
        
        # Training statistics
        self.loss_history = []
        self.accuracy_history = []
        
    def extract_features(self, token_idx, token_embedding, position, layer_idx, 
                         attention_scores=None, prev_tokens=None):
        """
        Extract features for prediction.
        
        Args:
            token_idx: Index of the token
            token_embedding: Embedding of the token
            position: Position of the token in the sequence
            layer_idx: Current layer index
            attention_scores: Optional attention scores for this token
            prev_tokens: Optional list of previous token indices
            
        Returns:
            features: Tensor of features for prediction
        """
        # Basic features
        features = [
            float(token_idx),
            float(position),
            float(layer_idx)
        ]
        
        # Add embedding statistics (mean, std, min, max)
        if token_embedding is not None:
            emb = token_embedding.detach().cpu().numpy().flatten()
            features.extend([
                float(np.mean(emb)),
                float(np.std(emb)),
                float(np.min(emb)),
                float(np.max(emb))
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
        
        # Add previous token information if available
        if prev_tokens is not None:
            for i, prev_idx in enumerate(prev_tokens[-5:]):  # Use last 5 tokens
                features.append(float(prev_idx))
        
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def predict(self, features):
        """
        Predict which KV blocks will be needed.
        
        Args:
            features: Input features tensor
            
        Returns:
            predictions: Tensor of probabilities for each KV block
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features.unsqueeze(0)).squeeze(0)
        return predictions
    
    def update(self, token_idx, k_value, v_value, features, is_hit=False):
        """
        Update the cache with a new token and record experience.
        
        Args:
            token_idx: Index of the token
            k_value: Key tensor for the token
            v_value: Value tensor for the token
            features: Features used for prediction
            is_hit: Whether this was a cache hit
            
        Returns:
            evicted: List of evicted token indices, or None if no eviction
        """
        # Record access
        self.access_history[token_idx] = self.access_history.get(token_idx, 0) + 1
        
        # Store features
        self.feature_history[token_idx] = features.detach().cpu().numpy()
        
        # Check if token is already in cache
        if token_idx in self.cache:
            # Cache hit, update values
            self.cache[token_idx] = (k_value, v_value)
            return None
        
        # Check if cache is full
        evicted = []
        if len(self.cache) >= self.max_cache_size:
            # Need to evict based on predictions
            all_indices = list(self.cache.keys())
            all_features = [torch.tensor(self.feature_history[idx], dtype=torch.float32, device=self.device) 
                           for idx in all_indices]
            
            if all_features:
                # Stack features and get predictions
                stacked_features = torch.stack(all_features)
                with torch.no_grad():
                    importance_scores = self.model(stacked_features).cpu().numpy()
                
                # Sort by importance (ascending)
                sorted_indices = np.argsort(importance_scores)
                
                # Evict least important token
                to_evict = all_indices[sorted_indices[0]]
                evicted.append(to_evict)
                del self.cache[to_evict]
            else:
                # If no features available, evict random token
                to_evict = random.choice(list(self.cache.keys()))
                evicted.append(to_evict)
                del self.cache[to_evict]
        
        # Add new token to cache
        self.cache[token_idx] = (k_value, v_value)
        
        return evicted if evicted else None
    
    def record_experience(self, features, target, reward):
        """
        Record experience for training.
        
        Args:
            features: Input features
            target: Target output (1 if needed, 0 if not)
            reward: Reward signal (positive if prediction was correct)
        """
        self.replay_buffer.append((features, target, reward))
    
    def train(self, num_batches=1):
        """
        Train the predictor using experiences from the replay buffer.
        
        Args:
            num_batches: Number of batches to train on
            
        Returns:
            avg_loss: Average loss during training
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # Not enough samples
        
        self.model.train()
        total_loss = 0.0
        
        for _ in range(num_batches):
            # Sample batch
            batch = random.sample(self.replay_buffer, self.batch_size)
            features, targets, rewards = zip(*batch)
            
            # Convert to tensors
            features = torch.stack([f.to(self.device) for f in features])
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features).squeeze()
            
            # Compute loss (weighted by rewards)
            loss = self.criterion(predictions, targets) * rewards.mean()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.loss_history.append(avg_loss)
        return avg_loss
    
    def get(self, token_idx):
        """
        Get key and value tensors for a token if it exists in the cache.
        
        Args:
            token_idx: Index of the token
            
        Returns:
            Tuple of (k_value, v_value) if token is in cache, None otherwise
        """
        if token_idx in self.cache:
            # Update access count
            self.access_history[token_idx] = self.access_history.get(token_idx, 0) + 1
            return self.cache[token_idx]
        return None
    
    def prefetch(self, current_tokens, features_list, k_cache, v_cache, num_tokens=10):
        """
        Prefetch tokens that are predicted to be needed soon.
        
        Args:
            current_tokens: List of current token indices
            features_list: List of feature tensors for prediction
            k_cache: Key cache
            v_cache: Value cache
            num_tokens: Number of tokens to prefetch
            
        Returns:
            prefetched: List of prefetched token indices
        """
        # Stack features and get predictions
        stacked_features = torch.stack(features_list)
        with torch.no_grad():
            importance_scores = self.model(stacked_features).cpu().numpy()
        
        # Get indices of tokens not in current_tokens
        all_indices = np.arange(len(importance_scores))
        mask = np.ones_like(all_indices, dtype=bool)
        for idx in current_tokens:
            if idx < len(mask):
                mask[idx] = False
        
        # Filter scores and get top-k
        filtered_scores = importance_scores[mask]
        filtered_indices = all_indices[mask]
        
        if len(filtered_indices) == 0:
            return []
        
        # Sort by importance (descending)
        sorted_idx = np.argsort(-filtered_scores)
        top_k_idx = sorted_idx[:min(num_tokens, len(sorted_idx))]
        prefetch_indices = filtered_indices[top_k_idx]
        
        # Prefetch tokens
        prefetched = []
        for idx in prefetch_indices:
            if idx < k_cache.shape[0] and idx not in self.cache:
                # Extract k and v values for this token
                k = k_cache[idx].clone()
                v = v_cache[idx].clone()
                
                # Add to cache
                self.cache[idx] = (k, v)
                prefetched.append(idx)
        
        return prefetched
    
    def evaluate(self, features, targets):
        """
        Evaluate the predictor on a set of features and targets.
        
        Args:
            features: List of feature tensors
            targets: List of target values (1 if needed, 0 if not)
            
        Returns:
            accuracy: Prediction accuracy
        """
        self.model.eval()
        with torch.no_grad():
            stacked_features = torch.stack(features)
            predictions = self.model(stacked_features).cpu().numpy()
            binary_preds = (predictions > 0.5).astype(int)
            targets = np.array(targets)
            accuracy = np.mean(binary_preds == targets)
        
        self.accuracy_history.append(accuracy)
        return accuracy
    
    def save_model(self, path):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history
        }, path)
    
    def load_model(self, path):
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['loss_history']
        self.accuracy_history = checkpoint['accuracy_history']


def select_kv_mlp(predictor, k_cache, v_cache, selected_indices):
    """
    Selects and aggregates KV caches using MLP predictor
    
    Args:
        predictor: KVBlockPredictor instance
        k_cache: Key cache (n, bh, d)
        v_cache: Value cache (n, bh, d)
        selected_indices: Indices of tokens to select from the cache
        
    Returns:
        selected_k: selected key cache
        selected_v: selected value cache
    """
    selected_k_list = []
    selected_v_list = []
    
    for idx in selected_indices:
        cache_entry = predictor.get(idx)
        if cache_entry is not None:
            k, v = cache_entry
            selected_k_list.append(k)
            selected_v_list.append(v)
    
    if not selected_k_list:
        return None, None
        
    selected_k = torch.stack(selected_k_list, dim=0)
    selected_v = torch.stack(selected_v_list, dim=0)
    
    return selected_k, selected_v
