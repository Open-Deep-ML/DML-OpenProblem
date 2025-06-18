import numpy as np

def dice_score(y_true, y_pred):
    """
    Calculate the Dice coefficient (also known as F1-score or Sørensen–Dice coefficient).
    
    Args:
        y_true (numpy.ndarray): Array of true values (ground truth)
        y_pred (numpy.ndarray): Array of predicted values
    
    Returns:
        float: Dice score value rounded to 3 decimal places
    """
    # Calculate intersection and sums
    intersection = np.logical_and(y_true, y_pred).sum()
    true_sum = y_true.sum()
    pred_sum = y_pred.sum()
    
    # Handle edge cases 
    if true_sum == 0 or pred_sum == 0:
        return 0.0
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection) / (true_sum + pred_sum)
    return round(float(dice), 3)

def test_dice_score():
    # Test case 1: Perfect overlap
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    expected_output = 1.000
    assert dice_score(y_true, y_pred) == expected_output, "Test case 1 failed"
    
    # Test case 2: No overlap
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    expected_output = 0.000
    assert dice_score(y_true, y_pred) == expected_output, "Test case 2 failed"
    
    # Test case 3: Partial overlap
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    expected_output = 0.667
    assert dice_score(y_true, y_pred) == expected_output, "Test case 3 failed"
    
    # Test case 4: All zeros
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    expected_output = 0.000
    assert dice_score(y_true, y_pred) == expected_output, "Test case 4 failed"
    
    # Test case 5: All ones
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    expected_output = 1.000
    assert dice_score(y_true, y_pred) == expected_output, "Test case 5 failed"
    
    # Test case 6: One empty, one full
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    expected_output = 0.000
    assert dice_score(y_true, y_pred) == expected_output, "Test case 6 failed"
    
    # Test case 7: Single element arrays
    y_true = np.array([1])
    y_pred = np.array([1])
    expected_output = 1.000
    assert dice_score(y_true, y_pred) == expected_output, "Test case 7 failed"
    
    # Test case 8: Different data types
    y_true = np.array([True, True, False, False])
    y_pred = np.array([1, 1, 0, 0])
    expected_output = 1.000
    assert dice_score(y_true, y_pred) == expected_output, "Test case 8 failed"
    

if __name__ == "__main__":
    test_dice_score()
    print("All Dice score test cases passed.")