from tinygrad.tensor import Tensor


def transpose_matrix_tg(a) -> Tensor:
    """
    Transpose a 2D matrix `a` using tinygrad.
    Inputs can be Python lists, NumPy arrays, or tinygrad Tensors.
    Returns a transposed Tensor.
    """
    a_t = Tensor(a)
    return a_t.transpose(0, 1)
