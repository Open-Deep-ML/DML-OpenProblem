from typing import List, Tuple


def fit_and_predict(X_train: List[float], y_train: List[float], X_test: List[float]) -> Tuple[float, float, List[float]]:
    n = len(X_train)
    x_mean = sum(X_train) / n
    y_mean = sum(y_train) / n

    num = 0.0
    den = 0.0
    for i in range(n):
        dx = X_train[i] - x_mean
        dy = y_train[i] - y_mean
        num += dx * dy
        den += dx * dx

    m = num / den if den != 0 else 0.0
    b = y_mean - m * x_mean

    y_pred = [m * x + b for x in X_test]
    return m, b, y_pred


