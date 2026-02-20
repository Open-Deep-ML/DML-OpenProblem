import torch


def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2×2 matrix using PyTorch.
    Input: 2×2 tensor; Output: 1-D tensor with the two eigenvalues in ascending order.
    """
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[1, 0]
    d = matrix[1, 1]
    trace = a + d
    det = a * d - b * c
    disc = trace * trace - 4 * det
    sqrt_disc = torch.sqrt(disc)
    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2
    eig = torch.stack([lambda1, lambda2])
    return torch.sort(eig).values
