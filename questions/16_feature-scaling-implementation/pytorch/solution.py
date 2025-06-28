import torch


def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize and Min-Max normalize input data using PyTorch.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = torch.as_tensor(data, dtype=torch.float)
    mean = data_t.mean(dim=0)
    std = data_t.std(dim=0, unbiased=False)
    standardized = (data_t - mean) / std
    min_val = data_t.min(dim=0).values
    max_val = data_t.max(dim=0).values
    normalized = (data_t - min_val) / (max_val - min_val)
    standardized = torch.round(standardized * 10000) / 10000
    normalized = torch.round(normalized * 10000) / 10000
    return standardized, normalized
