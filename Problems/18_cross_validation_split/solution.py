
import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True, random_seed=None):
    """
    Generate train and test indices for k-fold cross-validation.

    Parameters:
    X (np.ndarray): Feature dataset.
    y (np.ndarray): Target labels.
    k (int): Number of folds.
    shuffle (bool): Whether to shuffle the data before splitting.
    random_seed (int, optional): Seed for reproducibility.

    Returns:
    list of tuples: Each tuple contains train and test indices as lists.
    """
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.seed(random_seed) if random_seed is not None else None
        np.random.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1  # Distribute remainder among the first folds

    folds = []
    current = 0
    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size])
        current += fold_size

    return [(np.concatenate(folds[:i] + folds[i+1:]).tolist(), folds[i].tolist()) for i in range(k)]

def test_k_fold_cross_validation():
    # Test case 1: k=5, shuffle=False
    result = k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=5, shuffle=False)
    expected = [([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]), ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]),
                ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]), ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]),
                ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])]
    assert result == expected, "Test case 1 failed"

    # Test case 2: k=2, shuffle=True, fixed seed
    result = k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=2, shuffle=True, random_seed=42)
    expected = [([2, 9, 4, 3, 6], [8, 1, 5, 0, 7]), ([8, 1, 5, 0, 7], [2, 9, 4, 3, 6])]
    assert result == expected, "Test case 2 failed"

    # Test case 3: k=3, shuffle=False
    result = k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]),
                                     np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]), k=3, shuffle=False)
    expected = [([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4]),
                ([0, 1, 2, 3, 4, 10, 11, 12, 13, 14], [5, 6, 7, 8, 9]),
                ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14])]
    assert result == expected, "Test case 3 failed"

    # Test case 4: k=2, shuffle=False
    result = k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=2, shuffle=False)
    expected = [([5, 6, 7, 8, 9], [0, 1, 2, 3, 4]), ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])]
    assert result == expected, "Test case 4 failed"

    print("All k-fold cross-validation tests passed.")

if __name__ == "__main__":
    test_k_fold_cross_validation()
