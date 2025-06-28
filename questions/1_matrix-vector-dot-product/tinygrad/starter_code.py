from tinygrad.tensor import Tensor


def matrix_dot_vector_tg(a, b) -> Tensor:
    """
    Compute the product of matrix `a` and vector `b` using tinygrad.
    Inputs can be Python lists, NumPy arrays, or tinygrad Tensors.
    Returns a 1-D Tensor of length m, or Tensor(-1) if dimensions mismatch.
    """
    # Dimension mismatch check
    if len(a[0]) != len(b):
        return Tensor(-1)
    # Convert to Tensor
    Tensor(a)
    Tensor(b)
    # Your implementation here
    pass
