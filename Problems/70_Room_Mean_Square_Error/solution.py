import numpy as np
def rmse(y_true,y_pred):

    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays must have the same shape")
    if y_true.size == 0:
        raise ValueError("Arrays cannot be empty")
    
    return round(np.sqrt(np.mean((y_true - y_pred) ** 2)),3)
def test_rmse():
# Test Case 1: Normal Case  
    y_true1 = np.array([3, -0.5, 2, 7])
    y_pred1 = np.array([2.5, 0.0, 2, 8])
    expected1 = 0.612
    assert abs(rmse(y_true1, y_pred1)) == expected1, "Test Case Failed"

# Test Case 2: 2D Array 
    y_true2 = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred2 = np.array([[0, 2], [-1, 2], [8, -5]])
    expected2 = 0.842
    assert rmse(y_true2, y_pred2)==expected2, "Test Case Failed"

# Test Case 3: Perfect predictions
    y_true3 = np.array([[1, 2], [3, 4]])
    y_pred3 = np.array([[1, 2], [3, 4]])
    assert rmse(y_true3, y_pred3) == 0.0, "Test Case Failed"

# Test Case 4: Different shapes
    y_true4 = np.array([[1, 2], [3, 4]])
    y_pred4 = np.array([1, 2, 3, 4])
    try:
        rmse(y_true4, y_pred4)
        assert False, "Test Case Failed"
    except ValueError:
        pass

# Test Case 5: Empty arrays
    y_true5 = np.array([])
    y_pred5 = np.array([])
    try:
        rmse(y_true5, y_pred5)
        assert False, "Test Case Failed"
    except ValueError:
        pass

if __name__ == "__main__":
    test_rmse()
    print("All Test Cases Passed !")
