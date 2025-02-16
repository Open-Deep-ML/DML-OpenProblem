import numpy as np

def adjusted_r_squared(y_actual, y_predicted, num_predictors):
    """
    Calculate the Adjusted R-squared (R²_adj) value.

    :param y_actual: List or NumPy array of actual values
    :param y_predicted: List or NumPy array of predicted values
    :param num_predictors: Number of independent variables (features)
    :return: Adjusted R-squared value rounded to five decimal places
    """
    # Convert inputs to NumPy arrays
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # Get the number of observations (n)
    n = len(y_actual)

    # Compute R-squared (R²)
    y_mean = np.mean(y_actual)
    sse = np.sum((y_actual - y_predicted) ** 2)  # Sum of squared errors
    tss = np.sum((y_actual - y_mean) ** 2)  # Total sum of squares
    r_squared = 1 - (sse / tss)

    # Compute Adjusted R-squared (R²_adj)
    r2_adj = 1 - ((1 - r_squared) * (n - 1) / (n - num_predictors - 1))

    # Return the result rounded to five decimal places
    return round(r2_adj, 5)

def test_adjusted_r_squared():
    """
    Test cases for the adjusted_r_squared function.
    """
    # Test case 1: Standard case
    y_actual_1 = [3, -0.5, 2, 7, 4]
    y_pred_1 = [2.8, -0.3, 2.2, 6.9, 3.9]
    num_predictors_1 = 2
    expected_output_1 = 0.95902
    output_1 = adjusted_r_squared(y_actual_1, y_pred_1, num_predictors_1)
    assert output_1 == expected_output_1, f"Test case 1 failed: expected {expected_output_1}, got {output_1}"

    # Test case 2: Perfect prediction
    y_actual_2 = [3, 1, 4, 5, 6]
    y_pred_2 = [3, 1, 4, 5, 6]
    num_predictors_2 = 2
    expected_output_2 = 1.0
    output_2 = adjusted_r_squared(y_actual_2, y_pred_2, num_predictors_2)
    assert output_2 == expected_output_2, f"Test case 2 failed: expected {expected_output_2}, got {output_2}"

    # Test case 3: Poor prediction
    y_actual_3 = [3, 2, 1, 0]
    y_pred_3 = [10, 8, 6, 4]  # Completely off predictions
    num_predictors_3 = 1
    expected_output_3 = -5.5
    output_3 = adjusted_r_squared(y_actual_3, y_pred_3, num_predictors_3)
    assert output_3 == expected_output_3, f"Test case 3 failed: expected {expected_output_3}, got {output_3}"

    # Test case 4: R²_adj = 0 (random predictions)
    y_actual_4 = [1, 2, 3, 4, 5]
    y_pred_4 = [5, 4, 3, 2, 1]
    num_predictors_4 = 2
    expected_output_4 = -0.1
    output_4 = adjusted_r_squared(y_actual_4, y_pred_4, num_predictors_4)
    assert output_4 == expected_output_4, f"Test case 4 failed: expected {expected_output_4}, got {output_4}"

    print("All Adjusted R-squared test cases passed.")

if __name__ == "__main__":
    test_adjusted_r_squared()
