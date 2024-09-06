import numpy as np

def l1_regularization(X:np.array, y:np.array, alpha: float=0.1, learning_rate: float=0.01, max_iter: int=1000, tol: float=1e-4) -> tuple:
    n_samples, n_features = X.shape
    # zero out weights and biases
    weights = np.zeros(n_features)
    bias = 0

    for iteration in range(max_iter):
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y
        grad_w = (1 / n_samples) * np.dot(X.T, error) + alpha * np.sign(weights)
        grad_b = (1 / n_samples) * np.sum(error)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        if np.linalg.norm(grad_w, ord=1) < tol:
            break

    return weights, bias

def test_l1_regularization():

    X = np.array([[0,0], [1, 1], [2, 2]])
    y = np.array([0, 1, 2])
    alpha = 0.1
    l1_regularization(X,y , alpha = 0.1, max_iter = 5000)
    expected_weights = np.array([0.4250, 0.4250])
    expected_bias = 0.150
    weights, bias = l1_regularization(X, y, alpha=alpha, learning_rate=0.01, max_iter=1000)

    assert np.allclose(weights, expected_weights, atol=0.01), "Test case 1 failed"
    assert np.isclose(bias, expected_bias, atol=0.01), "Test case 1 failed"


if __name__ == "__main__":
    test_l1_regularization()
    print("All l1_regularization tests passed.")
