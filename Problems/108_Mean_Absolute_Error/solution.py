import numpy as np

def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between actual and predicted values.

    :param y_true: List of actual values
    :param y_pred: List of predicted values
    :return: Mean Absolute Error (MAE)
    """
    # Compute absolute differences
    absolute_errors = np.abs(np.array(y_true) - np.array(y_pred))
    # Compute mean absolute error and round to five decimal places
    mae = round(np.mean(absolute_errors), 5)
    return mae

def test_mean_absolute_error():
    # Test case 1: Basic MAE calculation
    y_true_1 = [3, -0.5, 2, 7]
    y_pred_1 = [2.5, 0.0, 2, 8]
    expected_output_1 = 0.5
    output_1 = mean_absolute_error(y_true_1, y_pred_1)
    assert output_1 == expected_output_1, \
        f"Test case 1 failed: expected {expected_output_1}, got {output_1}"

    # Test case 2: Higher differences
    y_true_2 = [1, 2, 3, 4]
    y_pred_2 = [2, 3, 4, 5]
    expected_output_2 = 1.0
    output_2 = mean_absolute_error(y_true_2, y_pred_2)
    assert output_2 == expected_output_2, \
        f"Test case 2 failed: expected {expected_output_2}, got {output_2}"

    # Test case 3: All exact matches
    y_true_3 = [5, 10, 15, 20]
    y_pred_3 = [5, 10, 15, 20]
    expected_output_3 = 0.0
    output_3 = mean_absolute_error(y_true_3, y_pred_3)
    assert output_3 == expected_output_3, \
        f"Test case 3 failed: expected {expected_output_3}, got {output_3}"

    # Test case 4: Large numbers
    y_true_4 = [100, 200, 300, 400]
    y_pred_4 = [110, 190, 310, 390]
    expected_output_4 = 10.0
    output_4 = mean_absolute_error(y_true_4, y_pred_4)
    assert output_4 == expected_output_4, \
        f"Test case 4 failed: expected {expected_output_4}, got {output_4}"

    # Test case 5: Negative values
    y_true_5 = [-5, -10, -15, -20]
    y_pred_5 = [-6, -9, -14, -19]
    expected_output_5 = 1.0
    output_5 = mean_absolute_error(y_true_5, y_pred_5)
    assert output_5 == expected_output_5, \
        f"Test case 5 failed: expected {expected_output_5}, got {output_5}"

    print("All Mean Absolute Error tests passed.")

if __name__ == "__main__":
    test_mean_absolute_error()
