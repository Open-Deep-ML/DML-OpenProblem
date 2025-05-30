import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """
    Find the best feature and threshold to split the dataset based on Gini impurity.

    :param X: Feature matrix of shape (n_samples, n_features)
    :param y: Labels array of shape (n_samples,), binary (0 or 1)
    :return: (feature_index, threshold) with lowest weighted Gini impurity
    """

    def gini_impurity(y_subset: np.ndarray) -> float:
        if len(y_subset) == 0:
            return 0.0
        p = np.mean(y_subset == 1)
        return 1.0 - (p ** 2 + (1 - p) ** 2)

    n_samples, n_features = X.shape
    best_feature = -1
    best_threshold = float('inf')
    best_gini = float('inf')

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_mask = X[:, feature_index] <= threshold
            right_mask = ~left_mask

            y_left, y_right = y[left_mask], y[right_mask]
            g_left, g_right = gini_impurity(y_left), gini_impurity(y_right)

            weighted_gini = (len(y_left) * g_left + len(y_right) * g_right) / n_samples

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold

def test():
    # Test 1: Balanced binary split
    X1 = np.array([[2.5], [3.5], [1.0], [4.0]])
    y1 = np.array([0, 1, 0, 1])
    f1, t1 = find_best_split(X1, y1)
    assert f1 == 0
    assert 1.0 <= t1 <= 3.5

    # Test 2: Pure set (Gini = 0)
    X2 = np.array([[1], [2], [3]])
    y2 = np.array([1, 1, 1])
    f2, t2 = find_best_split(X2, y2)
    assert f2 == 0
    assert t2 in [1, 2, 3]

    # Test 3: Alternating labels
    X3 = np.array([[1], [2], [3], [4]])
    y3 = np.array([0, 1, 0, 1])
    f3, t3 = find_best_split(X3, y3)
    assert f3 == 0
    assert t3 in [1, 2, 3, 4]

    # Test 4: No good split (non-separable)
    X4 = np.array([[1], [1], [1]])
    y4 = np.array([0, 1, 0])
    f4, t4 = find_best_split(X4, y4)
    assert f4 == 0
    assert t4 == 1

    # Test 5: Two features, first one irrelevant
    X5 = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
    y5 = np.array([0, 0, 1, 1])
    f5, t5 = find_best_split(X5, y5)
    assert f5 == 1
    assert t5 in [1, 2, 3, 4]

    # Test 6: Tiny dataset
    X6 = np.array([[1], [2]])
    y6 = np.array([0, 1])
    f6, t6 = find_best_split(X6, y6)
    assert f6 == 0
    assert t6 in [1, 2]

    print("All test cases passed.")

if __name__ == "__main__":
    test()
