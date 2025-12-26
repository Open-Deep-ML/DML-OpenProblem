import numpy as np


def bootstrap_sample(X, y):
    """Create a random bootstrap sample from the dataset."""
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]


def best_split(X, y):
    """Find the best split for a decision tree."""
    n_features = X.shape[1]
    best_feature, best_threshold, best_score = None, None, float("inf")

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] < threshold
            right_mask = ~left_mask

            if np.any(left_mask) and np.any(right_mask):
                left_mean, right_mean = np.mean(y[left_mask]), np.mean(y[right_mask])
                mse = np.mean((y[left_mask] - left_mean) ** 2) + np.mean((y[right_mask] - right_mean) ** 2)

                if mse < best_score:
                    best_feature, best_threshold, best_score = feature, threshold, mse

    return best_feature, best_threshold


def build_tree(X, y, depth=0, max_depth=5):
    """Recursively build a decision tree."""
    if depth >= max_depth or len(set(y)) == 1:
        return np.mean(y)

    feature, threshold = best_split(X, y)

    if feature is None:
        return np.mean(y)

    left_mask = X[:, feature] < threshold
    right_mask = ~left_mask

    return {
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(X[left_mask], y[left_mask], depth + 1, max_depth),
        "right": build_tree(X[right_mask], y[right_mask], depth + 1, max_depth),
    }


def predict_tree(tree, x):
    """Predict the output using a trained decision tree."""
    if not isinstance(tree, dict):
        return tree
    if x[tree["feature"]] < tree["threshold"]:
        return predict_tree(tree["left"], x)
    else:
        return predict_tree(tree["right"], x)


def random_forest(X, y, n_trees=5, max_depth=5):
    """Train a simple random forest."""
    trees = []
    for _ in range(n_trees):
        X_sample, y_sample = bootstrap_sample(X, y)
        tree = build_tree(X_sample, y_sample, max_depth=max_depth)
        trees.append(tree)
    return trees


def predict_forest(trees, X):
    """Predict the output using a trained random forest."""
    predictions = np.array([[predict_tree(tree, x) for tree in trees] for x in X])
    return np.mean(predictions, axis=1)  # Averaging for regression


def test_random_forest():
    np.random.seed(42)

    # Test case 1: Basic small dataset
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y_train = np.array([1, 3, 5, 7, 9])

    forest = random_forest(X_train, y_train, n_trees=3, max_depth=2)
    predictions = predict_forest(forest, X_train)

    assert len(predictions) == len(y_train), \
        f"Test case 1 failed: Expected {len(y_train)} predictions, got {len(predictions)}"

    # Test case 2: Random dataset with sine-cosine relation
    X_train = np.random.rand(10, 2) * 10
    y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])

    forest = random_forest(X_train, y_train, n_trees=5, max_depth=3)
    predictions = predict_forest(forest, X_train)

    assert len(predictions) == len(y_train), \
        f"Test case 2 failed: Expected {len(y_train)} predictions, got {len(predictions)}"

    # Test case 3: Single data point
    X_train = np.array([[5, 5]])
    y_train = np.array([10])

    forest = random_forest(X_train, y_train, n_trees=3, max_depth=2)
    predictions = predict_forest(forest, X_train)

    assert predictions[0] == 10, \
        f"Test case 3 failed: Expected 10, got {predictions[0]}"

    print("All random forest tests passed.")


if __name__ == "__main__":
    test_random_forest()
