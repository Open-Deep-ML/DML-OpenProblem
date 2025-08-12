import numpy as np


def l1_regularization_gradient_descent(
    X: np.array,
    y: np.array,
    alpha: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n_samples, n_features = X.shape

    np.zeros(n_features)
    # Your code here
    pass
