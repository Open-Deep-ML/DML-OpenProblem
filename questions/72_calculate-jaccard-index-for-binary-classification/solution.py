
import numpy as np

def jaccard_index(y_true, y_pred):
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    result = intersection / union
    if np.isnan(result):
        return 0.0
    return round(result, 3)
