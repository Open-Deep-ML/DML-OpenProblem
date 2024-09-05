import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    np.random.seed(seed)
    n_samples = np.shape(X)[0]
    # Concatenate X and y and do a random shuffle
    X_y = np.concatenate((X, y.reshape((len(y), 1))), axis=1)
    np.random.shuffle(X_y)
    subsets = []

    # Determine subsample size
    subsample_size = n_samples if replacements else n_samples // 2

    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_samples),
            size=subsample_size,
            replace=replacements)
        X_subset = X_y[idx][:, :-1]
        y_subset = X_y[idx][:, -1]
        subsets.append([X_subset, y_subset])
    return subsets

def test_get_random_subsets():
    # Test case 1
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    expected_output = [[[1, 2], [9, 10]], [1, 5]]
    assert np.array_equal(get_random_subsets(X, y, 1, False, seed=42), expected_output), "Test case 1 failed"
    
    # Test case 2
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([10, 20, 30, 40])
    expected_output = [[[1, 1], [3, 3], [2, 2], [2, 2]], [10, 30, 20, 20]]
    assert np.array_equal(get_random_subsets(X, y, 1, True, seed=42), expected_output), "Test case 2 failed"

if __name__ == "__main__":
    test_get_random_subsets()
    print("All get_random_subsets tests passed.")
