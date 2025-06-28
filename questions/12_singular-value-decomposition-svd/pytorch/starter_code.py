import torch


def svd_2x2_singular_values(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Approximate the SVD of a 2×2 matrix A using one Jacobi rotation.
    Returns (U, S, Vt) where S is a 1-D tensor of singular values.
    """
    # Your implementation here
    pass
