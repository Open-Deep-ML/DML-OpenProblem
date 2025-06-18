from tinygrad.tensor import Tensor

def feature_scaling_tg(data) -> tuple[Tensor, Tensor]:
    """
    Standardize and Min-Max normalize input data using tinygrad.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = Tensor(data).float()
    # Your implementation here
    pass
