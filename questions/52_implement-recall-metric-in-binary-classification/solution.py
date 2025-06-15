import numpy as np

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    try:
        return round(tp / (tp + fn), 3)
    except ZeroDivisionError:
        return 0.0
