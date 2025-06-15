import numpy as np

def f_score(y_true, y_pred, beta):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    op = precision * recall
    div = ((beta**2) * precision) + recall

    if div == 0 or op == 0:
        return 0

    score = op/div * (1 + (beta ** 2))
    return round(score, 3)

def test_f_score():
    # Test case 1
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    beta = 1
    expected_output = 0.857
    assert f_score(y_true, y_pred, beta) == expected_output, "Test case 1 failed"

    # Test case 2
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 0, 1])
    beta = 1
    expected_output = 0.4
    assert f_score(y_true, y_pred, beta) == expected_output, "Test case 2 failed"

    # Test case 3
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    beta = 2
    expected_output = 1
    assert f_score(y_true, y_pred, beta) == expected_output, "Test case 3 failed"

    # Test case 4
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1])
    beta = 2
    expected_output = 0.556
    assert f_score(y_true, y_pred, beta) == expected_output, "Test case 4 failed"

    # Test case 5
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    beta = 0.5
    expected_output = 0.0
    assert f_score(y_true, y_pred, beta) == expected_output, "Test case 5 failed"

    # Test case 6
    y_true = np.array([1, 0, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    beta = 0.5
    expected_output = 0.667
    assert f_score(y_true, y_pred, beta) == expected_output, "Test case 6 failed"

    # Test case 7
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    beta = 3
    expected_output = 0.0
    assert f_score(y_true, y_pred, beta) == expected_output, "Test case 7 failed"

    # Test case 8
    y_true = np.array([1, 0, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    beta = 0
    expected_output = 0.667
    assert f_score(y_true, y_pred, beta) == expected_output, "Test case 8 failed"

if __name__ == "__main__":
    test_f_score()
    print("All F-score tests passed.")
