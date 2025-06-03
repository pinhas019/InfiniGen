import torch
import numpy as np

class ARCCache:
    """
    Adaptive Replacement Cache (ARC) implementation for KV cache management.
    
    ARC maintains four lists:
    - T1: Recently used pages (recency)
    - B1: Ghost entries for recently evicted pages from T1
    - T2: Frequently used pages (frequency)
    - B2: Ghost entries for recently evicted pages from T2
    
    The algorithm adapts between recency and frequency based on workload patterns.
    """
    
    def __init__(self, max_size, device):
        """
        Initialize ARC cache.
        
        Args:
            max_size: Maximum number of entries in the cache (T1 + T2)
            device: Torch device for storing tensors
        """
        self.max_size = max_size
        self.device = device
        
        # Parameter p balances between recency (T1) and frequency (T2)
        self.p = 0
        
        # Initialize lists
        self.T1 = {}  # Recently used pages
        self.T2 = {}  # Frequently used pages
        self.B1 = {}  # Ghost entries for recently evicted pages from T1
        self.B2 = {}  # Ghost entries for recently evicted pages from T2
        
        # Mapping from token indices to their location (T1 or T2)
        self.location = {}
        
        # Tracking access history for each token
        self.access_count = {}
        
    def size(self):
        """Return current size of the cache (T1 + T2)."""
        return len(self.T1) + len(self.T2)
        
    def _replace(self, token_idx):
        """
        Choose a victim for replacement based on ARC algorithm.
        
        Args:
            token_idx: Index of the token to be added
        
        Returns:
            Index of the token to be evicted
        """
        if len(self.T1) > 0 and (len(self.T1) > self.p or 
                                (token_idx in self.B2 and len(self.T1) == self.p)):
            # Evict from T1 (recency)
            victim = next(iter(self.T1))
            self.B1[victim] = True
            del self.T1[victim]
            return victim
        else:
            # Evict from T2 (frequency)
            victim = next(iter(self.T2))
            self.B2[victim] = True
            del self.T2[victim]
            return victim
    
    def _adjust_p(self, token_idx):
        """
        Adjust the balance parameter p based on cache hits in ghost lists.
        
        Args:
            token_idx: Index of the accessed token
        """
        if token_idx in self.B1:
            # Hit in B1 (recency ghost list) - increase p
            self.p = min(self.p + 1, self.max_size)
        elif token_idx in self.B2:
            # Hit in B2 (frequency ghost list) - decrease p
            self.p = max(self.p - 1, 0)
    
    def update(self, token_idx, k_value, v_value):
        """
        Update the cache with a new token or access an existing one.
        
        Args:
            token_idx: Index of the token
            k_value: Key tensor for the token
            v_value: Value tensor for the token
            
        Returns:
            Tuple of (evicted_idx, evicted_k, evicted_v) if eviction occurred,
            or None if no eviction
        """
        # Check if token is already in cache
        if token_idx in self.location:
            # Cache hit
            location = self.location[token_idx]
            
            # Update access count
            self.access_count[token_idx] = self.access_count.get(token_idx, 0) + 1
            
            if location == "T1":
                # Move from T1 to T2 (recency to frequency)
                k, v = self.T1.pop(token_idx)
                self.T2[token_idx] = (k, v)
                self.location[token_idx] = "T2"
            else:  # location == "T2"
                # Already in T2, update values
                self.T2[token_idx] = (k_value, v_value)
                
            return None  # No eviction
            
        # Token not in cache, check ghost lists and adjust p
        self._adjust_p(token_idx)
        
        # Remove from ghost lists if present
        if token_idx in self.B1:
            del self.B1[token_idx]
        if token_idx in self.B2:
            del self.B2[token_idx]
        
        # Check if cache is full
        if self.size() >= self.max_size:
            # Need to evict
            victim_idx = self._replace(token_idx)
            victim_location = self.location[victim_idx]
            
            if victim_location == "T1":
                victim_k, victim_v = self.T1[victim_idx]
            else:  # victim_location == "T2"
                victim_k, victim_v = self.T2[victim_idx]
                
            del self.location[victim_idx]
            del self.access_count[victim_idx]
            
            # Add new token to T1 (recency)
            self.T1[token_idx] = (k_value, v_value)
            self.location[token_idx] = "T1"
            self.access_count[token_idx] = 1
            
            return (victim_idx, victim_k, victim_v)
        else:
            # Cache not full, add to T1
            self.T1[token_idx] = (k_value, v_value)
            self.location[token_idx] = "T1"
            self.access_count[token_idx] = 1
            
            return None  # No eviction
    
    def get(self, token_idx):
        """
        Get key and value tensors for a token if it exists in the cache.
        
        Args:
            token_idx: Index of the token
            
        Returns:
            Tuple of (k_value, v_value) if token is in cache, None otherwise
        """
        location = self.location.get(token_idx)
        
        if location is None:
            return None
            
        if location == "T1":
            k, v = self.T1[token_idx]
            
            # Move from T1 to T2 (recency to frequency)
            self.T1.pop(token_idx)
            self.T2[token_idx] = (k, v)
            self.location[token_idx] = "T2"
            
            # Update access count
            self.access_count[token_idx] = self.access_count.get(token_idx, 0) + 1
            
            return (k, v)
        else:  # location == "T2"
            # Update access count
            self.access_count[token_idx] = self.access_count.get(token_idx, 0) + 1
            
            return self.T2[token_idx]
    
    def select_tokens(self, num_tokens):
        """
        Select the most important tokens based on access frequency.
        
        Args:
            num_tokens: Number of tokens to select
            
        Returns:
            List of token indices sorted by importance
        """
        # Sort tokens by access count (descending)
        sorted_tokens = sorted(self.access_count.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        
        # Return top N token indices
        return [idx for idx, _ in sorted_tokens[:num_tokens]]
