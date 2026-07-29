import numpy as np

def pinball_loss(y_true, y_pred, quantile=0.5):
    """
    Calculate the mean Pinball (Quantile) Loss between two arrays.

    Parameters:
    y_true (numpy.ndarray): Array of true values
    y_pred (numpy.ndarray): Array of predicted values
    quantile (float): Target quantile in the open interval (0, 1)

    Returns:
    float: Mean Pinball Loss rounded to 3 decimal places
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays must have the same shape")
    if y_true.size == 0:
        raise ValueError("Arrays cannot be empty")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in the open interval (0, 1)")

    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)
    return round(float(np.mean(loss)), 3)
