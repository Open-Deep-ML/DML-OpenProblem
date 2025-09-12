from tinygrad.tensor import Tensor

def matrix_dot_vector_tg(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the product of matrix `a` and vector `b` using tinygrad.
    Inputs will be tinygrad Tensors.
    Returns a 1-D Tensor of length m, or Tensor(-1) if dimensions mismatch.
    """
    if len(a[0]) != len(b):
        return Tensor(-1)
    return a @ b