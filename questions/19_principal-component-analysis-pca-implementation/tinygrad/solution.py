import numpy as np
from tinygrad.tensor import Tensor


def pca_tg(data, k) -> Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tinygrad Tensor.
    Input: list, NumPy array, or Tensor of shape (n_samples, n_features).
    Returns: a Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    """
    arr = np.array(data, dtype=float)
    # Standardize
    arr_std = (arr - arr.mean(axis=0)) / arr.std(axis=0)
    # Covariance
    cov = np.cov(arr_std, rowvar=False)
    # Eigen decomposition
    vals, vecs = np.linalg.eig(cov)
    idx = np.argsort(vals)[::-1]
    pcs = vecs[:, idx[:k]]
    pcs = np.round(pcs, 4)
    return Tensor(pcs)
