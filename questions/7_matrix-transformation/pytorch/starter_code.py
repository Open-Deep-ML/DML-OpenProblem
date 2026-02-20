import torch


def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    torch.as_tensor(A, dtype=torch.float)
    torch.as_tensor(T, dtype=torch.float)
    torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    pass
