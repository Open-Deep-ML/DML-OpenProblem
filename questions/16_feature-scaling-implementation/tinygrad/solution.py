import numpy as np
from tinygrad.tensor import Tensor

def feature_scaling_tg(data) -> tuple[Tensor, Tensor]:
    """
    Standardize and Min-Max normalize input data using tinygrad.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = Tensor(data).float()
    data_np = data_t.numpy()
    mean = np.mean(data_np, axis=0)
    std = np.std(data_np, axis=0)
    standardized_np = (data_np - mean) / std
    min_val = np.min(data_np, axis=0)
    max_val = np.max(data_np, axis=0)
    normalized_np = (data_np - min_val) / (max_val - min_val)
    return Tensor(np.round(standardized_np, 4)), Tensor(np.round(normalized_np, 4))
