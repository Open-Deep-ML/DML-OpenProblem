from tinygrad.tensor import Tensor

def calculate_matrix_mean_tg(matrix, mode: str) -> Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using tinygrad.
    Inputs can be Python lists, NumPy arrays, or tinygrad Tensors.
    Returns a 1-D Tensor of means or raises ValueError on invalid mode.
    """
    v_t = Tensor(matrix).float()
    # Your implementation here
    pass
