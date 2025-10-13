import torch
from typing import List

def log_softmax_stable(x: torch.Tensor) -> torch.Tensor:
    """
    Compute log-softmax of x in a numerically stable way.
    Args:
        x: torch.Tensor of shape (N,) or (N, C)
    Returns:
        log-softmax tensor of same shape
    """
    # Your code here
    pass

def cross_entropy_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss given true labels and predicted logits.
    Args:
        y_true: torch.Tensor of shape (N,)
        y_pred: torch.Tensor of shape (N, C)
    Returns:
        scalar mean loss
    """
    # Your code here
    pass
