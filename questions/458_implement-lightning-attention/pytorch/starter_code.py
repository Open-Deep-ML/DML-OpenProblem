import torch

def lightning_attention(Q, K, V, decay):
    """
    Implement causal linear attention with exponential decay.

    Args:
        Q (torch.Tensor): Query tensor of shape (seq_len, head_dim).
        K (torch.Tensor): Key tensor of shape (seq_len, head_dim).
        V (torch.Tensor): Value tensor of shape (seq_len, head_dim).
        decay (float): Exponential decay factor (lambda), between 0 and 1.

    Returns:
        torch.Tensor: Output tensor of shape (seq_len, head_dim).
    """
    pass
