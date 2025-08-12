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
