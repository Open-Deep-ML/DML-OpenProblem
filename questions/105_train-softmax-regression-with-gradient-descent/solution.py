import numpy as np


def train_softmaxreg(
    X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int
) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for softmax regression, that collects mean-reduced
    CE losses, accuracies.
    Returns
    -------
    B : list[float]
        CxM updated parameter vector rounded to 4 floating points
    losses : list[float]
        collected values of a Cross Entropy rounded to 4 floating points
    """

    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def accuracy(y_pred, y_true):
        return (np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)).sum() / len(
            y_true
        )

    def ce_loss(y_pred, y_true):
        true_labels_idx = np.argmax(y_true, axis=1)
        return -np.sum(np.log(y_pred)[list(range(len(y_pred))), true_labels_idx])

    y = y.astype(int)
    C = y.max() + 1  # we assume that classes start from 0
    y = np.eye(C)[y]
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    B = np.zeros((X.shape[1], C))
    accuracies, losses = [], []

    for epoch in range(iterations):
        y_pred = softmax(X @ B)
        B -= learning_rate * X.T @ (y_pred - y)
        losses.append(round(ce_loss(y_pred, y), 4))
        accuracies.append(round(accuracy(y_pred, y), 4))

    return B.T.round(4).tolist(), losses
