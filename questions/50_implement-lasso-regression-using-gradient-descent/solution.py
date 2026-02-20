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
    # Zero out weights and bias
    weights = np.zeros(n_features)
    bias = 0

    for iteration in range(max_iter):
        # Predict values
        y_pred = np.dot(X, weights) + bias
        # Calculate error
        error = y_pred - y
        # Gradient for weights with L1 penalty
        grad_w = (1 / n_samples) * np.dot(X.T, error) + alpha * np.sign(weights)
        # Gradient for bias (no penalty for bias)
        grad_b = (1 / n_samples) * np.sum(error)

        # Update weights and bias
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        # Check for convergence
        if np.linalg.norm(grad_w, ord=1) < tol:
            break

    return weights, bias
