import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    y = y.reshape(-1, 1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    B = np.zeros((X.shape[1], 1))
    losses = []

    for _ in range(iterations):
        y_pred = sigmoid(X @ B)
        B -= learning_rate * X.T @ (y_pred - y)
        loss = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        losses.append(round(loss, 4))

    return B.flatten().round(4).tolist(), losses
