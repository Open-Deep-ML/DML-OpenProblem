import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    np.random.seed(seed)

    n, m = X.shape
    
    subset_size = n if replacements else n // 2
    idx = np.array([np.random.choice(n, subset_size, replace=replacements) for _ in range(n_subsets)])
    # convert all ndarrays to lists
    return [(X[idx][i].tolist(), y[idx][i].tolist()) for i in range(n_subsets)]


def test_get_random_subsets():
    # Test case 1
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    expected_output = [([[3, 4], [9, 10]], [2, 5]), ([[7, 8], [3, 4]], [4, 2]), ([[3, 4], [1, 2]], [2, 1])]
    assert get_random_subsets(X, y, 3, False, seed=42) == expected_output, "Test case 1 failed"

    
    # Test case 2
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([10, 20, 30, 40])
    expected_output = [([[3, 3], [4, 4], [1, 1], [3, 3]], [30, 40, 10, 30])]
    assert get_random_subsets(X, y, 1, True, seed=42) == expected_output, "Test case 2 failed"

if __name__ == "__main__":
    test_get_random_subsets()
    print("All get_random_subsets tests passed.")
