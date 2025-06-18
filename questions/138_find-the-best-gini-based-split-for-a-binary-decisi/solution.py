import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    def gini(y_subset: np.ndarray) -> float:
        if y_subset.size == 0:
            return 0.0
        p = y_subset.mean()
        return 1.0 - (p**2 + (1 - p)**2)

    n_samples, n_features = X.shape
    best_feature, best_threshold = -1, float('inf')
    best_gini = float('inf')

    for f in range(n_features):
        for threshold in np.unique(X[:, f]):
            left = y[X[:, f] <= threshold]
            right = y[X[:, f] > threshold]
            g_left, g_right = gini(left), gini(right)
            weighted = (len(left) * g_left + len(right) * g_right) / n_samples
            if weighted < best_gini:
                best_gini, best_feature, best_threshold = weighted, f, threshold

    return best_feature, best_threshold
