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
