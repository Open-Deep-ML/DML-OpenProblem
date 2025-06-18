import numpy as np

def conjugate_gradient(A: np.array, b: np.array, n: int, x0: np.array=None, tol=1e-8) -> np.array:

    # calculate initial residual vector
    x = np.zeros_like(b)
    r = residual(A, b, x) # residual vector
    rPlus1 = r
    p = r # search direction vector

    for i in range(n):

        # line search step value - this minimizes the error along the current search direction
        alp = alpha(A, r, p)

        # new x and r based on current p (the search direction vector)
        x = x + alp * p
        rPlus1 = r - alp * (A@p)

        # calculate beta - this ensures that all vectors are A-orthogonal to each other
        bet = beta(r, rPlus1)

        # update x and r
        # using a othogonal search direction ensures we get all the information we need in more direction and then don't have to search in that direction again
        p = rPlus1 + bet * p

        # update residual vector
        r = rPlus1

        # break if less than tolerance
        if np.linalg.norm(residual(A,b,x)) < tol:
            break

    return x

def residual(A: np.array, b: np.array, x: np.array) -> np.array:
    # calculate linear system residuals
    return b - A @ x

def alpha(A: np.array, r: np.array, p: np.array) -> float:

    # calculate step size
    alpha_num = np.dot(r, r)
    alpha_den = np.dot(p @ A, p)

    return alpha_num/alpha_den

def beta(r: np.array, r_plus1: np.array) -> float:

    # calculate direction scaling
    beta_num = np.dot(r_plus1, r_plus1)
    beta_den = np.dot(r, r)

    return beta_num/beta_den
