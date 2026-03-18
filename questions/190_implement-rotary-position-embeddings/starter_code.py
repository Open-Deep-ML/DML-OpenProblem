import numpy as np

# Implement your function below.
def apply_rotary_embeddings(q, k, positions, dim, base=10000.0):
    """
    Apply Rotary Position Embeddings (RoPE) to query and key tensors.

    Args:
        q (np.ndarray): Query array of shape (seq_len, head_dim).
        k (np.ndarray): Key array of shape (seq_len, head_dim).
        positions (np.ndarray): 1D integer array of token positions, length seq_len.
        dim (int): Number of dimensions to apply rotation to (must be even, <= head_dim).
        base (float): Base frequency for computing rotation angles.

    Returns:
        tuple: (q_rotated, k_rotated) arrays of same shape as inputs.
    """
    pass
