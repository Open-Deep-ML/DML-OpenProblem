import numpy as np

def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

def test_precision():
    # Test case 1
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    expected_output = 1.0
    assert precision(y_true, y_pred) == expected_output, "Test case 1 failed"
    
    # Test case 2
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 0, 1])
    expected_output = 0.5
    assert precision(y_true, y_pred) == expected_output, "Test case 2 failed"

if __name__ == "__main__":
    test_precision()
    print("All precision tests passed.")
