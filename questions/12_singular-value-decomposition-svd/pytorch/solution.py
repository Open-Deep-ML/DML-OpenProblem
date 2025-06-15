import torch


def svd_2x2_singular_values(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Approximate the SVD of a 2×2 matrix A using one Jacobi rotation.
    Returns (U, S, Vt) where S is a 1-D tensor of singular values.
    """
    # compute AᵀA
    a2 = A.transpose(0, 1) @ A
    # initialize V
    V = torch.eye(2, dtype=A.dtype, device=A.device)
    # Jacobi rotation angle
    if torch.isclose(a2[0, 0], a2[1, 1]):
        theta = torch.tensor(torch.pi / 4, dtype=A.dtype, device=A.device)
    else:
        theta = 0.5 * torch.atan2(2 * a2[0, 1], a2[0, 0] - a2[1, 1])
    c = torch.cos(theta)
    s = torch.sin(theta)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    # diagonalize
    D = R.transpose(0, 1) @ a2 @ R
    V = V @ R
    # singular values
    S = torch.sqrt(torch.tensor([D[0, 0], D[1, 1]], dtype=A.dtype, device=A.device))
    # compute U
    S_inv = torch.diag(1.0 / S)
    U = A @ V @ S_inv
    return U, S, V.transpose(0, 1)
