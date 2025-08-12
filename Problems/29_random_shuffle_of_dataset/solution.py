import numpy as np

def shuffle_data(X, y, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    return X_shuffled, y_shuffled

def test_shuffle_data() -> None:
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4]) 
    s=42
    expected_X = np.array( [[3, 4], [7, 8], [1, 2], [5, 6]])
    expected_y = np.array([2, 4, 1, 3])
    X_shuffled, y_shuffled=shuffle_data(X,y,s)
    assert np.array_equal(X_shuffled, expected_X) 
    assert np.array_equal(y_shuffled, expected_y)
    
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([10, 20, 30, 40])
    s=24
    expected_X = np.array([[4, 4],[2, 2],[1, 1],[3, 3]])
    expected_y = np.array([40, 20, 10, 30])
    X_shuffled, y_shuffled=shuffle_data(X,y,s)
    assert np.array_equal(X_shuffled, expected_X) 
    assert np.array_equal(y_shuffled, expected_y) 
    
if __name__ == "__main__":
    test_shuffle_data()
    print("All test_shuffle_data tests passed.")