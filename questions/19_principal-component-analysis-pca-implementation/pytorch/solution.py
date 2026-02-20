import torch


def pca(data, k) -> torch.Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    """
    data_t = torch.as_tensor(data, dtype=torch.float)
    # Standardize
    mean = data_t.mean(dim=0, keepdim=True)
    std = data_t.std(dim=0, unbiased=False, keepdim=True)
    data_std = (data_t - mean) / std
    # Covariance
    cov = torch.cov(data_std.T)
    # Eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eig(cov)
    vals = eigenvalues.real
    vecs = eigenvectors.real
    # Sort by descending eigenvalue
    idx = torch.argsort(vals, descending=True)
    pcs = vecs[:, idx[:k]]
    # Round to 4 decimals
    pcs = torch.round(pcs * 10000) / 10000
    return pcs
