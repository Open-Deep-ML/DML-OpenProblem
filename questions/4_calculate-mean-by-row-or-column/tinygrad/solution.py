from tinygrad.tensor import Tensor


def calculate_matrix_mean_tg(matrix, mode: str) -> Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using tinygrad.
    Inputs can be Python lists, NumPy arrays, or tinygrad Tensors.
    Returns a 1-D Tensor of means or raises ValueError on invalid mode.
    """
    v_t = Tensor(matrix).float()
    n_obs = v_t.shape[1]
    n_feat = v_t.shape[0]
    if mode == "column":
        return v_t.sum(axis=1) / n_obs
    elif mode == "row":
        return v_t.sum(axis=0) / n_feat
    else:
        raise ValueError("Mode must be 'row' or 'column'")
