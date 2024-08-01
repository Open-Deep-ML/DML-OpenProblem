import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Convert the inputs to numpy arrays
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Compute the normal equation
    X_transpose = X.T
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
    
    # Round the result to four decimal places
    theta = np.round(theta, 4).flatten().tolist()
    
    return theta

def test_linear_regression_normal_equation() -> None:
    # Test case 1
    X = [[1, 1], [1, 2], [1, 3]]
    y = [1, 2, 3]
    assert linear_regression_normal_equation(X, y) == [-0.0, 1.0], "Test case 1 failed"

    # Test case 2
    X = [[1, 3, 4], [1, 2, 5], [1, 3, 2]]
    y = [1, 2, 1]
    assert linear_regression_normal_equation(X, y) == [4.0, -1.0, -0.0], "Test case 2 failed"

if __name__ == "__main__":
    test_linear_regression_normal_equation()
    print("All linear_regression_normal_equation tests passed.")
