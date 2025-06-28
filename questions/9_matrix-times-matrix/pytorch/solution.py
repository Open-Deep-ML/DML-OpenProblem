import torch


def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    a_t = torch.as_tensor(a)
    b_t = torch.as_tensor(b)
    if a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)
    return a_t.matmul(b_t)
