import torch


def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Compute inverse of a 2×2 matrix using PyTorch.
    Input can be Python list, NumPy array, or torch Tensor.
    Returns a 2×2 tensor or None if the matrix is singular.
    """
    m = torch.as_tensor(matrix, dtype=torch.float)
    a, b = m[0, 0], m[0, 1]
    c, d = m[1, 0], m[1, 1]
    det = a * d - b * c
    if det == 0:
        return None
    inv = torch.stack(
        [torch.stack([d / det, -b / det]), torch.stack([-c / det, a / det])]
    )
    return inv
