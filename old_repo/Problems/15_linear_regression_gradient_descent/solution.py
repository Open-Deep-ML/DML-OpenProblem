import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    # Number of training examples and features
    m, n = X.shape
    
    # Initialize the coefficients (weights) to zeros
    theta = np.zeros((n, 1))
    
    # Perform gradient descent for a given number of iterations
    for _ in range(iterations):
        # Compute the predictions
        predictions = X @ theta
        
        # Calculate the error (difference between predictions and actual values)
        errors = predictions - y.reshape(-1, 1)
        
        # Calculate the updates for the coefficients
        updates = X.T @ errors / m
        
        # Update the coefficients
        theta -= alpha * updates
    
    # Round the coefficients to four decimal places and return as a 1D array
    return np.round(theta.flatten(), 4)

def test_linear_regression_gradient_descent() -> None:
    # Test case 1
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([1, 2, 3])
    alpha = 0.01
    iterations = 1000
    assert np.allclose(linear_regression_gradient_descent(X, y, alpha, iterations), np.array([0.1107, 0.9513]), atol=1e-4), "Test case 1 failed"

if __name__ == "__main__":
    test_linear_regression_gradient_descent()
    print("All linear_regression_gradient_descent tests passed.")
