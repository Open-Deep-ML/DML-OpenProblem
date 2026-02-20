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


def test_train_softmaxreg():
    # Test 1
    X = np.array(
        [
            [2.52569869, 2.33335813, 1.77303921, 0.41061103, -1.66484491],
            [1.51013861, 1.30237106, 1.31989315, 1.36087958, 0.46381252],
            [-2.09699866, -1.35960405, -1.04035503, -2.25481082, -0.32359947],
            [-0.96660088, -0.60680633, -0.72017167, -1.73257187, -1.12811486],
            [-0.38096611, -0.24852455, 0.18789426, 0.52359424, 1.30725962],
            [0.54828787, 0.33156614, 0.10676247, 0.30694669, -0.37555384],
            [-3.03393135, -2.01966141, -0.6546858, -0.90330912, 2.89185791],
            [0.28602304, -0.1265, -0.52209915, 0.28309144, -0.5865882],
            [-0.26268117, 0.76017979, 1.84095557, -0.23245038, 1.80716891],
            [0.30283562, -0.40231495, -1.29550644, -0.1422727, -1.78121713],
        ]
    )
    y = np.array([2, 3, 0, 0, 1, 3, 0, 1, 2, 1])
    learning_rate = 3e-2
    iterations = 10
    expected_b = [
        [-0.0841, -0.5693, -0.3651, -0.2423, -0.5344, 0.0339],
        [0.2566, 0.0535, -0.2104, -0.4004, 0.2709, -0.1461],
        [-0.1318, 0.2109, 0.3998, 0.523, -0.1001, 0.0545],
        [-0.0407, 0.3049, 0.1757, 0.1197, 0.3637, 0.0576],
    ]
    expected_losses = [
        13.8629,
        10.7201,
        9.3163,
        8.4942,
        7.9132,
        7.4598,
        7.0854,
        6.7653,
        6.4851,
        6.2358,
    ]
    b, ce = train_softmaxreg(X, y, learning_rate, iterations)
    assert b == expected_b and ce == expected_losses, "Test case 1 failed"

    # Test 2
    X = np.array(
        [
            [-0.55605887, -0.74922526, -0.1913345, 0.41584056],
            [-1.05481124, -1.13763371, -1.28685937, -1.0710115],
            [-1.17111877, -1.46866663, -0.75898143, 0.15915148],
            [-1.21725723, -1.55590285, -0.69318542, 0.3580615],
            [-1.90316075, -2.06075824, -2.2952422, -1.87885386],
            [-0.79089629, -0.98662696, -0.52955027, 0.07329079],
            [1.97170638, 2.65609694, 0.6802377, -1.47090364],
            [1.46907396, 1.61396429, 1.69602021, 1.29791351],
            [0.03095068, 0.15148081, -0.34698116, -0.74306029],
            [-1.40292946, -1.99308861, -0.1478281, 1.72332995],
        ]
    )
    y = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    learning_rate = 1e-2
    iterations = 7
    expected_b = [
        [-0.0052, 0.0148, 0.0562, -0.113, -0.2488],
        [0.0052, -0.0148, -0.0562, 0.113, 0.2488],
    ]
    expected_losses = [6.9315, 6.4544, 6.0487, 5.7025, 5.4055, 5.1493, 4.9269]
    b, ce = train_softmaxreg(X, y, learning_rate, iterations)
    assert b == expected_b and ce == expected_losses, "Test case 2 failed"

    print("All tests passed")


if __name__ == "__main__":
    test_train_softmaxreg()
