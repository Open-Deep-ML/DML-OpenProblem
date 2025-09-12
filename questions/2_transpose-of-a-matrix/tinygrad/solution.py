from tinygrad.tensor import Tensor

def transpose_matrix_tg(a) -> Tensor:
    """
    Transpose a 2D matrix `a` using tinygrad.
    Inputs are tinygrad Tensors.
    Returns a transposed Tensor.
    """
    return a.T
