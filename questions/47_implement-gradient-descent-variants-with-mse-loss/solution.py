import numpy as np

def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    learning_rate: float,
    n_iterations: int,
    batch_size: int = 1,
    method: str = "batch",
) -> np.ndarray:
    m: int = X.shape[0]
    n: int = X.shape[1]
    w = np.zeros((n, 1))

    match method:
        case "batch":
            batch_size: int = m
        case "stochastic":
            batch_size: int = 1
        case "mini_batch":
            batch_size: int = batch_size
        case _:
            return w

    for _ in range(n_iterations):
        for i in range(0, m, batch_size):
            x_batch = X[i : min(i + batch_size, m), :]
            y_batch = y[i : min(i + batch_size, m)]
            y_hat = x_batch @ w
            derivative = x_batch.T @ (y_hat.reshape((-1, 1)) - y_batch.reshape((-1, 1)))
            w = w - 2 * learning_rate / batch_size * derivative
    return w.flatten()
