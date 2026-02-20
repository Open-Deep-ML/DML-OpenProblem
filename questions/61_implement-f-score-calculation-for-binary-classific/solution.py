import numpy as np


def f_score(y_true, y_pred, beta):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    op = precision * recall
    div = ((beta**2) * precision) + recall

    if div == 0 or op == 0:
        return 0.0

    score = (1 + (beta**2)) * op / div
    return round(score, 3)
