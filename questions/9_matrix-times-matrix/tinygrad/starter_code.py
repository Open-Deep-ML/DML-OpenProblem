from tinygrad.tensor import Tensor

def matrixmul_tg(a, b) -> Tensor:
    """
    Multiply two matrices using tinygrad.
    Inputs can be Python lists, NumPy arrays, or tinygrad Tensors.
    Returns a 2D Tensor of shape (m, n) or a scalar Tensor -1.0 if dimensions mismatch.
    """
    # dimension mismatch
    if len(a[0]) != len(b):
        return Tensor(-1.0)
    # convert and multiply
    a_t = Tensor(a)
    b_t = Tensor(b)
    return a_t.matmul(b_t)
