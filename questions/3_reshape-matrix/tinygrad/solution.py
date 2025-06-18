from tinygrad.tensor import Tensor

def reshape_matrix_tg(a, new_shape) -> Tensor:
    """
    Reshape a 2D matrix `a` to shape `new_shape` using tinygrad.
    Inputs can be Python lists, NumPy arrays, or tinygrad Tensors.
    Returns a Tensor of shape `new_shape`, or an empty Tensor on mismatch.
    """
    # Dimension check
    if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
        return Tensor([])
    a_t = Tensor(a)
    return a_t.reshape(new_shape)
