import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha):
    loss = np.mean((y_true - X@w)**2) + alpha * np.sum(w**2)

    return np.array(loss)


def test_ridge_loss():
    # Test case 1
    X = np.array([[1,1],[2,1],[3,1],[4,1]])
    W = np.array([.2,2])
    y = np.array([2,3,4,5])
    
    expected = 2.204
    assert np.array_equal(expected, ridge_loss(X,W,y,.1))


    # Test case 2
    X = np.array([[1,1,4],[2,1,2],[3,1,.1],[4,1,1.2],[1,2,3]])
    W = np.array([.2,2,5])
    y = np.array([2,3,4,5,2])
    
    expected = 161.7884
    assert np.array_equal(expected, ridge_loss(X,W,y,.1))

  


if __name__ == "__main__":
    test_ridge_loss()
    print("All ridge_loss tests passed.")
