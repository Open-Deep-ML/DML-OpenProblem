import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    if torch.det(T_t) == 0 or torch.det(S_t) == 0:
        return torch.tensor(-1.)
    T_inv = torch.inverse(T_t)
    out = T_inv @ A_t @ S_t
    return torch.round(out * 1000) / 1000
