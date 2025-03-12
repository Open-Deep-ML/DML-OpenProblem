import numpy as np


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    n: int,
    x0: np.ndarray | None = None,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """

    def residual(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        return b - A @ x

    # calculate initial residual vector
    x = np.zeros_like(b) if x0 is None else x0
    r = residual(A, b, x)  # residual vector
    r_plus = r
    p = r  # search direction vector

    for _ in range(n):
        # line search step value - this minimizes the error along the current search direction
        alpha = (r.T @ r) / (p.T @ A @ p)

        # new x and r based on current p (the search direction vector)
        x = x + alpha * p

        r_plus = r - alpha * (A @ p)

        # break if less than tolerance
        if np.linalg.norm(r_plus) < tol:
            break

        # calculate beta - this ensures that all vectors are A-orthogonal to each other
        beta = (r_plus.T @ r_plus) / (r.T @ r)

        # update x and r
        # using a othogonal search direction ensures we get all the information
        # we need in more direction and then don't have to search in that direction again
        p = r_plus + beta * p

        # update residual vector
        r = r_plus

    return x
