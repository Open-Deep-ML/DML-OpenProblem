import numpy as np

def r_squared(y_actual, y_predicted):
    """
    Calculate the R-squared (R²) value to measure how well the model fits the data.

    :param y_actual: List or NumPy array of actual values
    :param y_predicted: List or NumPy array of predicted values
    :return: R-squared value rounded to five decimal places
    """
    # Convert inputs to NumPy arrays for computation
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # Compute the mean of actual values
    y_mean = np.mean(y_actual)

    # Calculate Sum of Squared Errors (SSE)
    sse = np.sum((y_actual - y_predicted) ** 2)

    # Calculate Total Sum of Squares (TSS)
    tss = np.sum((y_actual - y_mean) ** 2)

    # Compute R²
    r2 = 1 - (sse / tss)

    # Return rounded value
    return round(r2, 5)

def test_r_squared():
    """
    Test cases for the r_squared function.
    """
    # Test case 1: Perfect fit (R² = 1)
    y_actual_1 = [3, -0.5, 2, 7]
    y_pred_1 = [3, -0.5, 2, 7]  # Perfect prediction
    expected_output_1 = 1.0
    output_1 = r_squared(y_actual_1, y_pred_1)
    assert output_1 == expected_output_1, f"Test case 1 failed: expected {expected_output_1}, got {output_1}"

    # Test case 2: Moderate fit
    y_actual_2 = [3, -0.5, 2, 7]
    y_pred_2 = [2.5, 0.0, 2, 8]
    expected_output_2 = 0.94861
    output_2 = r_squared(y_actual_2, y_pred_2)
    assert output_2 == expected_output_2, f"Test case 2 failed: expected {expected_output_2}, got {output_2}"

    # Test case 3: Poor fit
    y_actual_3 = [3, 2, 1, 0]
    y_pred_3 = [10, 8, 6, 4]  # Completely off predictions
    expected_output_3 = -3.16667
    output_3 = r_squared(y_actual_3, y_pred_3)
    assert output_3 == expected_output_3, f"Test case 3 failed: expected {expected_output_3}, got {output_3}"

    # Test case 4: R² = 0 (random predictions)
    y_actual_4 = [1, 2, 3, 4, 5]
    y_pred_4 = [5, 4, 3, 2, 1]  # Inverted predictions
    expected_output_4 = 0.0
    output_4 = r_squared(y_actual_4, y_pred_4)
    assert output_4 == expected_output_4, f"Test case 4 failed: expected {expected_output_4}, got {output_4}"

    print("All R-squared test cases passed.")

if __name__ == "__main__":
    test_r_squared()
