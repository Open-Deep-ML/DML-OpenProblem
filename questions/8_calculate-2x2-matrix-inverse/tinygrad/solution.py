from tinygrad.tensor import Tensor


def inverse_2x2_tg(matrix) -> Tensor | None:
    """
    Compute inverse of a 2×2 matrix using tinygrad.
    Input can be Python list, NumPy array, or tinygrad Tensor.
    Returns a 2×2 Tensor or None if the matrix is singular.
    """
    m = Tensor(matrix).float()
    a, b = m[0, 0], m[0, 1]
    c, d = m[1, 0], m[1, 1]
    det = a * d - b * c
    if det.numpy() == 0:
        return None
    inv = Tensor([[d, -b], [-c, a]]) / det
    return inv
