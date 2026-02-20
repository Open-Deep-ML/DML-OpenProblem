from tinygrad.tensor import Tensor


def transform_matrix_tg(A, T, S) -> Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S for 2×2 matrices using tinygrad.
    Inputs A, T, S can be Python lists, NumPy arrays, or tinygrad Tensors.
    Returns a 2×2 Tensor or Tensor(-1.) if T or S is singular.
    """
    Tensor(A).float()
    Tensor(T).float()
    Tensor(S).float()
    # Your implementation here
    pass
