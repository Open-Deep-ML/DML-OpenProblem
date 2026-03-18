import numpy as np

# Implement your class and function below.
class KVCache:
    def __init__(self):
        """Initialize an empty KV cache."""
        pass

    def update(self, new_k, new_v):
        """
        Append new keys and values to the cache.

        Args:
            new_k (np.ndarray): New key array of shape (new_seq_len, head_dim).
            new_v (np.ndarray): New value array of shape (new_seq_len, head_dim).

        Returns:
            tuple: (full_keys, full_values) including all cached entries.
        """
        pass

    def get(self):
        """
        Return the current cached keys and values.

        Returns:
            tuple: (keys, values) or (None, None) if cache is empty.
        """
        pass

def attention_with_kv_cache(q, k, v, cache):
    """
    Compute scaled dot-product attention using a KV cache.

    Args:
        q (np.ndarray): Query array of shape (seq_len, head_dim).
        k (np.ndarray): New key array of shape (new_seq_len, head_dim).
        v (np.ndarray): New value array of shape (new_seq_len, head_dim).
        cache (KVCache): The KV cache to update and use.

    Returns:
        tuple: (output, cache) where output has shape (seq_len, head_dim).
    """
    pass
