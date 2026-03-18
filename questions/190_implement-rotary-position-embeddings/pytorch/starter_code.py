import torch

def apply_rotary_embeddings(q, k, positions, dim, base=10000.0):
    """
    Apply Rotary Position Embeddings (RoPE) to query and key tensors.

    Args:
        q (torch.Tensor): Query tensor of shape (seq_len, head_dim).
        k (torch.Tensor): Key tensor of shape (seq_len, head_dim).
        positions (torch.Tensor): 1D integer tensor of token positions.
        dim (int): Number of dimensions to apply rotation to (must be even, <= head_dim).
        base (float): Base frequency for computing rotation angles.

    Returns:
        tuple: (q_rotated, k_rotated) tensors of same shape as inputs.
    """
    pass
