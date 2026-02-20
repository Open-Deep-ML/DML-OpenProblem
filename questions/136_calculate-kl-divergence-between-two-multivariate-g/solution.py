import numpy as np


def multivariate_kl_divergence(
    mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray
) -> float:
    def trace(x: np.ndarray) -> float:
        return np.diag(x).sum()

    p = Cov_p.shape[0]
    return float(
        1
        / 2
        * (
            np.log(np.linalg.det(Cov_q) / np.linalg.det(Cov_p))
            - p
            + (mu_p - mu_q).T @ np.linalg.inv(Cov_q) @ (mu_p - mu_q)
            + trace(np.linalg.inv(Cov_q) @ Cov_p)
        )
    )
