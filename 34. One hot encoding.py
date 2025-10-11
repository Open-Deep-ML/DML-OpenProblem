import torch
from typing import Optional

def to_categorical(x: torch.Tensor, n_col: Optional[int] = None) -> torch.Tensor:
    """
    Perform one-hot encoding on a 1D integer tensor `x`. If `n_col` is not provided, infer it from the max value in `x`.
    """
    if n_col is None:
        n_col = x.max().item() + 1

    x = x.to(torch.long)
    one_hot = torch.nn.functional.one_hot(x, num_classes=n_col)
    one_hot = one_hot.float()

    return one_hot
