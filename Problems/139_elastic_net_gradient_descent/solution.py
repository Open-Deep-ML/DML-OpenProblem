import numpy as np


def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    """
    Implement Elastic Net regression using gradient descent.

    Parameters:
    X: Feature matrix (n_samples, n_features)
    y: Target values (n_samples,)
    alpha1: L1 regularization strength (Lasso)
    alpha2: L2 regularization strength (Ridge)
    learning_rate: Step size for gradient descent
    max_iter: Maximum number of iterations
    tol: Convergence tolerance

    Returns:
    tuple: (weights, bias)
    """
    n_samples, n_features = X.shape

    # Initialize weights and bias
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(max_iter):
        # Make predictions
        y_pred = np.dot(X, weights) + bias

        # Calculate residuals
        error = y_pred - y

        # Calculate gradients
        grad_w = (1 / n_samples) * np.dot(X.T, error) + alpha1 * np.sign(weights) + 2 * alpha2 * weights
        grad_b = (1 / n_samples) * np.sum(error)

        # Update weights and bias
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        # Check for convergence
        if np.linalg.norm(grad_w, ord=1) < tol:
            break

    return weights, bias


def test_elastic_net_gradient_descent():
    """Test cases for Elastic Net implementation"""

    # Test case 1: Simple linear relationship
    X = np.array([[0, 0], [1, 1], [2, 2]])
    y = np.array([0, 1, 2])
    alpha1, alpha2 = 0.1, 0.1

    weights, bias = elastic_net_gradient_descent(
        X, y, alpha1=alpha1, alpha2=alpha2, learning_rate=0.01, max_iter=1000
    )

    expected_weights = np.array([0.37, 0.37])
    expected_bias = 0.24

    assert np.allclose(weights, expected_weights, atol=0.05), "Test case 1 failed"
    assert np.isclose(bias, expected_bias, atol=0.05), "Test case 1 failed"

    # Test case 2: More complex relationship
    X = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4, 5])
    alpha1, alpha2 = 0.1, 0.1

    weights, bias = elastic_net_gradient_descent(
        X, y, alpha1=alpha1, alpha2=alpha2, learning_rate=0.01, max_iter=2000
    )

    expected_weights = np.array([0.42, 0.48])
    expected_bias = 0.68

    assert np.allclose(weights, expected_weights, atol=0.05), "Test case 2 failed"
    assert np.isclose(bias, expected_bias, atol=0.05), "Test case 2 failed"


if __name__ == "__main__":
    test_elastic_net_gradient_descent()
    print("All Elastic Net tests passed!")
