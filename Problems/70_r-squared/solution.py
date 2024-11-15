import numpy as np

def r_squared(y_true, y_pred):
    """
    Calculate the R-squared (R²) coefficient of determination.
    
    Args:
        y_true (numpy.ndarray): Array of true values
        y_pred (numpy.ndarray): Array of predicted values
    
    Returns:
        float: R-squared value rounded to 3 decimal places
    """
    if np.array_equal(y_true, y_pred):
        return 1.0
    
    # Calculate mean of true values
    y_mean = np.mean(y_true)
    
    # Calculate Sum of Squared Residuals (SSR)
    ssr = np.sum((y_true - y_pred) ** 2)
    
    # Calculate Total Sum of Squares (SST)
    sst = np.sum((y_true - y_mean) ** 2)

    try:
        # Calculate R-squared
        r2 = 1 - (ssr / sst)
        if np.isinf(r2):
            return 0.0
        return round(r2, 3)
    except ZeroDivisionError:
        return 0.0

def test_r_squared():
    # Test case 1: Perfect prediction
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    expected_output = 1.000
    assert r_squared(y_true, y_pred) == expected_output, "Test case 1 failed"

    # Test case 2: Good prediction
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    expected_output = 0.989
    assert r_squared(y_true, y_pred) == expected_output, "Test case 2 failed"

    # Test case 3: Poor prediction
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([2, 1, 4, 3, 5])
    expected_output = 0.600
    output = r_squared(y_true, y_pred)
    assert r_squared(y_true, y_pred) == expected_output, "Test case 3 failed"

    # Test case 4: Worst possible prediction (predicting mean)
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([3, 3, 3, 3, 3])
    expected_output = 0.000
    assert r_squared(y_true, y_pred) == expected_output, "Test case 4 failed"
    
    # Test case 5
    y_true = np.array([3, 3, 3, 3, 3])
    y_pred = np.array([1, 2, 3, 4, 5])
    expected_output = 0.000
    assert r_squared(y_true, y_pred) == expected_output, "Test case 5 failed"

    # Test case 6: Negative R-squared (predictions worse than mean)
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 4, 3, 2, 1])
    expected_output = -3.000
    assert r_squared(y_true, y_pred) == expected_output, "Test case 6 failed"

    # Test case 7: All zeros 
    y_true = np.array([0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0])
    expected_output = 1.000
    assert r_squared(y_true, y_pred) == expected_output, "Test case 7 failed"
    
    # Test case 8 : output = -inf
    y_true = np.array([-2, -2, -2])
    y_pred = np.array([-2, -2, -2 + 1e-8])
    expected_output = 0.000
    assert r_squared(y_true, y_pred) == expected_output, "Test case 8 failed"

if __name__ == "__main__":
    test_r_squared()
    print("All R-squared tests passed.")