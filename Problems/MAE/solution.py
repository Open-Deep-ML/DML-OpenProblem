import numpy as np

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between two arrays.
    
    Parameters:
    y_true (numpy.ndarray): Array of true values
    y_pred (numpy.ndarray): Array of predicted values
    
    Returns:
    float: Mean Absolute Error rounded to 3 decimal places
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays must have the same shape")
    if y_true.size == 0:
        raise ValueError("Arrays cannot be empty")
    
    return round(np.mean(np.abs(y_true - y_pred)), 3)

def test_mae():
    # Test Case 1: Normal Case
    y_true1 = np.array([3, -0.5, 2, 7])
    y_pred1 = np.array([2.5, 0.0, 2, 8])
    expected1 = 0.500
    assert mae(y_true1, y_pred1) == expected1, "Test Case 1 Failed"

    # Test Case 2: 2D Array
    y_true2 = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred2 = np.array([[0, 2], [-1, 2], [8, -5]])
    expected2 = 0.750
    assert mae(y_true2, y_pred2) == expected2, "Test Case 2 Failed"

    # Test Case 3: Perfect predictions
    y_true3 = np.array([[1, 2], [3, 4]])
    y_pred3 = np.array([[1, 2], [3, 4]])
    assert mae(y_true3, y_pred3) == 0.0, "Test Case 3 Failed"

    # Test Case 4: Different shapes
    y_true4 = np.array([[1, 2], [3, 4]])
    y_pred4 = np.array([1, 2, 3, 4])
    try:
        mae(y_true4, y_pred4)
        assert False, "Test Case 4 Failed: Should raise ValueError"
    except ValueError:
        pass

    # Test Case 5: Empty arrays
    y_true5 = np.array([])
    y_pred5 = np.array([])
    try:
        mae(y_true5, y_pred5)
        assert False, "Test Case 5 Failed: Should raise ValueError"
    except ValueError:
        pass

    # Test Case 6: Arrays with negative values
    y_true6 = np.array([-1, -2, -3])
    y_pred6 = np.array([-1.5, -2.2, -2.8])
    expected6 = 0.300
    assert mae(y_true6, y_pred6) == expected6, "Test Case 6 Failed"

    # Test Case 7: Mixed positive and negative differences
    y_true7 = np.array([1, -1, 0])
    y_pred7 = np.array([-1, 1, 0])
    expected7 = 1.333
    assert mae(y_true7, y_pred7) == expected7, "Test Case 7 Failed"

if __name__ == "__main__":
    test_mae()
    print("All Test Cases Passed!")