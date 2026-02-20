import numpy as np


def dice_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    true_sum = y_true.sum()
    pred_sum = y_pred.sum()

    # Handle edge cases
    if true_sum == 0 or pred_sum == 0:
        return 0.0

    dice = (2.0 * intersection) / (true_sum + pred_sum)
    return round(float(dice), 3)
