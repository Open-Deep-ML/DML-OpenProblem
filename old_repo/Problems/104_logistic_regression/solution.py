import numpy as np


def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    z = np.dot(X, weights) + bias
    z = np.clip(z, -500, 500)  # Prevent overflow in exp
    probabilities = 1 / (1 + np.exp(-z))
    return (probabilities >= 0.5).astype(int)


def test_predict_logistic():
    # Test case 1: Simple linearly separable case
    X1 = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]])
    w1 = np.array([1, 1])
    b1 = 0
    expected1 = np.array([1, 1, 0, 0])
    assert np.array_equal(predict_logistic(X1, w1, b1), expected1), "Test case 1 failed"

    # Test case 2: Decision boundary case
    X2 = np.array([[0, 0], [0.1, 0.1], [-0.1, -0.1]])
    w2 = np.array([1, 1])
    b2 = 0
    expected2 = np.array([1, 1, 0])
    assert np.array_equal(predict_logistic(X2, w2, b2), expected2), "Test case 2 failed"

    # Test case 3: Higher dimensional input
    X3 = np.array([[1, 2, 3], [-1, -2, -3], [0.5, 1, 1.5]])
    w3 = np.array([0.1, 0.2, 0.3])
    b3 = -1
    expected3 = np.array([1, 0, 0])
    assert np.array_equal(predict_logistic(X3, w3, b3), expected3), "Test case 3 failed"

    #     # Test case 4: Single feature
    X4 = np.array([[1], [2], [-1], [-2]]).reshape(-1, 1)
    w4 = np.array([2])
    b4 = 0
    expected4 = np.array([1, 1, 0, 0])
    assert np.array_equal(predict_logistic(X4, w4, b4), expected4), "Test case 4 failed"

    #     # Test case 5: Numerical stability test with large values
    X6 = np.array([[1000, 2000], [-1000, -2000]])
    w6 = np.array([0.1, 0.1])
    b6 = 0
    result6 = predict_logistic(X6, w6, b6)
    assert result6[0] == 1 and result6[1] == 0, "Test case 5 failed"


if __name__ == "__main__":
    test_predict_logistic()
    print("All test cases passed!")
