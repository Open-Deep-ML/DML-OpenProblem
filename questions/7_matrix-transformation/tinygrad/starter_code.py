from tinygrad.tensor import Tensor

def transform_matrix_tg(A, T, S) -> Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S for 2×2 matrices using tinygrad.
    Inputs A, T, S can be Python lists, NumPy arrays, or tinygrad Tensors.
    Returns a 2×2 Tensor or Tensor(-1.) if T or S is singular.
    """
    A_t = Tensor(A).float()
    T_t = Tensor(T).float()
    S_t = Tensor(S).float()
    # Your implementation here
    pass
