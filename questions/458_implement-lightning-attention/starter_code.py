import numpy as np

# Implement your function below.
def lightning_attention(Q, K, V, decay):
    """
    Implement causal linear attention with exponential decay.

    Args:
        Q (np.ndarray): Query array of shape (seq_len, head_dim).
        K (np.ndarray): Key array of shape (seq_len, head_dim).
        V (np.ndarray): Value array of shape (seq_len, head_dim).
        decay (float): Exponential decay factor (lambda), between 0 and 1.

    Returns:
        np.ndarray: Output array of shape (seq_len, head_dim).
    """
    pass
