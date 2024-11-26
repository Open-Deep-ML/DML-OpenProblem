import numpy as np


def recall(y_true, y_pred):
    denom = sum(y_true)
    if denom == 0:
        return 0.0
    return round(y_pred.dot(y_true)/denom, 3)


def test_recall():
    # Test case 1
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    expected_output = 0.75
    assert recall(y_true, y_pred) == expected_output, "Test case 1 failed"

    # Test case 2
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 0, 1])
    expected_output = 0.333
    assert recall(y_true, y_pred) == expected_output, "Test case 2 failed"

    # Test case 3
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    expected_output = 1
    assert recall(y_true, y_pred) == expected_output, "Test case 3 failed"

    # Test case 4
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1])
    expected_output = 0.5
    assert recall(y_true, y_pred) == expected_output, "Test case 4 failed"

    # Test case 5
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    expected_output = 0
    assert recall(y_true, y_pred) == expected_output, "Test case 5 failed"

    # Test case 6
    y_true = np.array([1, 0, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    expected_output = 0.667
    assert recall(y_true, y_pred) == expected_output, "Test case 6 failed"


if __name__ == "__main__":
    test_recall()
    print("All recall tests passed.")