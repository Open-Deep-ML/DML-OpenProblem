from tinygrad.tensor import Tensor


def inverse_2x2_tg(matrix) -> Tensor | None:
    """
    Compute inverse of a 2×2 matrix using tinygrad.
    Input can be Python list, NumPy array, or tinygrad Tensor.
    Returns a 2×2 Tensor or None if the matrix is singular.
    """
    Tensor(matrix).float()
    # Your implementation here
    pass
