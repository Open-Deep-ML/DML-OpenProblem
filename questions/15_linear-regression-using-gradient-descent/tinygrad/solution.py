import numpy as np
from tinygrad.tensor import Tensor

def linear_regression_gradient_descent_tg(X, y, alpha, iterations) -> Tensor:
    """
    Solve linear regression via gradient descent using tinygrad autograd.
    X: Tensor or convertible shape (m,n); y: shape (m,) or (m,1).
    alpha: learning rate; iterations: number of steps.
    Returns a 1-D Tensor of length n, rounded to 4 decimals.
    """
    X_t = Tensor(X).float()
    y_t = Tensor(y).float().reshape(-1,1)
    m, n = X_t.shape
    theta = Tensor([[0.0] for _ in range(n)])
    for _ in range(iterations):
        preds = X_t.matmul(theta)
        loss = (preds - y_t).pow(2).mean()
        loss.backward()
        # gradient step
        theta = theta - alpha * theta.grad
        theta.grad = None
    res = theta.reshape(n).detach().numpy()
    return Tensor(np.round(res, 4))
