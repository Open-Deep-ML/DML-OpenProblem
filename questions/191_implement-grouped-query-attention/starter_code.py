import numpy as np

# Implement your function below.
def grouped_query_attention(Q, K, V, num_kv_heads):
    """
    Implement Grouped Query Attention.

    Args:
        Q (np.ndarray): Query tensor of shape (num_q_heads, seq_len, head_dim).
        K (np.ndarray): Key tensor of shape (num_kv_heads, seq_len, head_dim).
        V (np.ndarray): Value tensor of shape (num_kv_heads, seq_len, head_dim).
        num_kv_heads (int): Number of key-value heads.

    Returns:
        np.ndarray: Attention output of shape (num_q_heads, seq_len, head_dim).
    """
    pass
