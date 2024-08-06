import numpy as np

def log_softmax(scores: np.ndarray) -> np.ndarray:
    # Subtract the maximum value for numerical stability
    scores = scores - np.max(scores)
    return np.round(scores - np.log(np.sum(np.exp(scores))), 4)

def test_log_softmax():
    # Test case 1
    A = np.array([1,2,3])
    assert np.allclose(log_softmax(A), [-2.4076, -1.4076, -0.4076])

    # Test case 2
    A = np.array([1,1,1])
    assert np.allclose(log_softmax(A), [-1.0986, -1.0986, -1.0986])

    # Test case 3
    A = np.array([1,1,0])
    assert np.allclose(log_softmax(A), [-0.862, -0.862, -1.862])

if __name__ == "__main__":
    test_log_softmax()
    print("All log softmax tests passed.")
