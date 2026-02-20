import numpy as np
from tinygrad.tensor import Tensor


def linear_regression_normal_equation_tg(X, y) -> Tensor:
    """
    Solve linear regression via the normal equation using tinygrad.
    X: list, NumPy array, or Tensor of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D Tensor of length n, rounded to 4 decimals.
    """
    X_np = np.array(X, dtype=float)
    y_np = np.array(y, dtype=float).reshape(-1, 1)
    theta = np.linalg.inv(X_np.T.dot(X_np)).dot(X_np.T).dot(y_np)
    theta = np.round(theta.flatten(), 4)
    return Tensor(theta)
