import numpy as np
from tinygrad.tensor import Tensor


def determinant_4x4_tg(matrix) -> Tensor:
    """
    Compute the determinant of a 4×4 matrix using tinygrad.
    Input can be a Python list, NumPy array, or tinygrad Tensor of shape (4,4).
    Returns a 0-D Tensor containing the determinant.
    """
    # convert to NumPy array
    if isinstance(matrix, Tensor):
        arr = matrix.numpy()
    else:
        arr = np.array(matrix, dtype=float)
    det = float(np.linalg.det(arr))
    return Tensor(det)
