from tinygrad.tensor import Tensor


def calculate_covariance_matrix_tg(vectors) -> Tensor:
    """
    Calculate the covariance matrix for given feature vectors using tinygrad.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a Tensor of shape (n_features, n_features).
    """
    v_t = Tensor(vectors).float()
    n_features, n_obs = v_t.shape
    # compute feature means
    means = v_t.sum(axis=1).reshape(n_features, 1) / n_obs
    centered = v_t - means
    cov = centered.matmul(centered.transpose(0, 1)) / (n_obs - 1)
    return cov
