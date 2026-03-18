import torch

def grouped_query_attention(Q, K, V, num_kv_heads):
    """
    Implement Grouped Query Attention.

    Args:
        Q (torch.Tensor): Query tensor of shape (num_q_heads, seq_len, head_dim).
        K (torch.Tensor): Key tensor of shape (num_kv_heads, seq_len, head_dim).
        V (torch.Tensor): Value tensor of shape (num_kv_heads, seq_len, head_dim).
        num_kv_heads (int): Number of key-value heads.

    Returns:
        torch.Tensor: Attention output of shape (num_q_heads, seq_len, head_dim).
    """
    pass
