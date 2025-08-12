import torch


def pca(data, k) -> torch.Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    """
    torch.as_tensor(data, dtype=torch.float)
    # Your implementation here
    pass
