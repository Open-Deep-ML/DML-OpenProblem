import numpy as np

def jaccard_index(y_true, y_pred):
    """
    Calculate the Jaccard Index between two binary arrays.
    Jaccard Index = |intersection| / |union|
    """
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    result = intersection / union
    if np.isnan(result):
        return 0.0
    return round(result, 3)

def test_jaccard():
    # Test case 1: Perfect match
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0, 1])
    expected_output = 1.0
    assert jaccard_index(y_true, y_pred) == expected_output, "Test case 1 failed"

    # Test case 2: No overlap
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    expected_output = 0.0
    assert jaccard_index(y_true, y_pred) == expected_output, "Test case 2 failed"

    # Test case 3: Partial overlap
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 0])
    expected_output = 0.5
    assert jaccard_index(y_true, y_pred) == expected_output, "Test case 3 failed"

    # Test case 4: More complex partial overlap
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 0])
    expected_output = 0.5
    assert jaccard_index(y_true, y_pred) == expected_output, "Test case 4 failed"

    # Test case 5: Small overlap
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0])
    expected_output = 0.167
    assert jaccard_index(y_true, y_pred) == expected_output, "Test case 5 failed"

    # Test case 6: Edge case - all zeros
    y_true = np.array([0, 0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    expected_output = 0.0
    assert jaccard_index(y_true, y_pred) == expected_output, "Test case 6 failed"

if __name__ == "__main__":
    test_jaccard()
    print("All Jaccard index tests passed.")