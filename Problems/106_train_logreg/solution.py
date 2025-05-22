import numpy as np


def train_logreg(X: np.ndarray, y: np.ndarray,
                 learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression, that collects sum-reduced
    BCE losses, accuracies. Assigns label "0" if the P(x_i)<=0.5 and "1" otherwise.

    Returns
    -------
    B : list[float]
        1xM updated parameter vector rounded to 4 floating points
    losses : list[float]
        collected values of a BCE loss function (LLF) rounded to 4 floating points
    """

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def accuracy(y_pred, y_true):
        return (y_true == np.rint(y_pred)).sum() / len(y_true)

    def bce_loss(y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    y = y.reshape(-1, 1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    B = np.zeros((X.shape[1], 1))
    accuracies, losses = [], []

    for epoch in range(iterations):
        y_pred = sigmoid(X @ B)
        B -= learning_rate * X.T @ (y_pred - y)
        losses.append(round(bce_loss(y_pred, y), 4))
        accuracies.append(round(accuracy(y_pred, y), 4))

    return B.flatten().round(4).tolist(), losses


def test_train_logreg():
    # Test 1
    X = np.array([[ 0.76743473, -0.23413696, -0.23415337,  1.57921282],
       [-1.4123037 ,  0.31424733, -1.01283112, -0.90802408],
       [-0.46572975,  0.54256004, -0.46947439, -0.46341769],
       [-0.56228753, -1.91328024,  0.24196227, -1.72491783],
       [-1.42474819, -0.2257763 ,  1.46564877,  0.0675282 ],
       [ 1.85227818, -0.29169375, -0.60063869, -0.60170661],
       [ 0.37569802,  0.11092259, -0.54438272, -1.15099358],
       [ 0.19686124, -1.95967012,  0.2088636 , -1.32818605],
       [ 1.52302986, -0.1382643 ,  0.49671415,  0.64768854],
       [-1.22084365, -1.05771093, -0.01349722,  0.82254491]])
    y = np.array([1., 0., 0., 0., 1., 1., 0., 0., 1., 0.])
    learning_rate = 1e-3
    iterations = 10
    b, llf = train_logreg(X, y, learning_rate, iterations)
    assert b == [-0.0097, 0.0286, 0.015, 0.0135, 0.0316] and \
        llf == [6.9315, 6.9075, 6.8837, 6.8601, 6.8367, 6.8134, 6.7904, 6.7675, 6.7448, 6.7223], \
            'Test case 1 failed'

    # Test 2
    X = np.array([[ 0.76743473,  1.57921282, -0.46947439],
       [-0.23415337,  1.52302986, -0.23413696],
       [ 0.11092259, -0.54438272, -1.15099358],
       [-0.60063869,  0.37569802, -0.29169375],
       [-1.91328024,  0.24196227, -1.72491783],
       [-1.01283112, -0.56228753,  0.31424733],
       [-0.1382643 ,  0.49671415,  0.64768854],
       [-0.46341769,  0.54256004, -0.46572975],
       [-1.4123037 , -0.90802408,  1.46564877],
       [ 0.0675282 , -0.2257763 , -1.42474819]])
    y = np.array([1., 1., 0., 0., 0., 0., 1., 1., 0., 0.])
    learning_rate = 1e-1
    iterations = 10
    b, llf = train_logreg(X, y, learning_rate, iterations)
    assert b == [-0.2509, 0.9325, 1.6218, 0.6336] and \
        llf == [6.9315, 5.5073, 4.6382, 4.0609, 3.6503, 3.3432, 3.1045, 2.9134, 2.7567, 2.6258], \
            'Test case 2 failed'

    print('All tests passed')


if __name__ == '__main__':
    test_train_logreg()