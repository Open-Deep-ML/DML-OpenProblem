from tinygrad.tensor import Tensor


def scalar_multiply_tg(matrix, scalar) -> Tensor:
    """
    Multiply each element of a 2D matrix by a scalar using tinygrad.
    Inputs can be Python lists, NumPy arrays, or tinygrad Tensors.
    Returns a 2D Tensor of the same shape.
    """
    # Convert input to Tensor
    Tensor(matrix)
    # Your implementation here
    pass
