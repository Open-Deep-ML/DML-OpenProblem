import torch

def rmsnorm(x, weight, eps=1e-6):
    """
    Apply Root Mean Square Layer Normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, features).
        weight (torch.Tensor): Learnable scale parameter of shape (features,).
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: RMSNorm-normalized tensor of same shape as x.
    """
    pass
