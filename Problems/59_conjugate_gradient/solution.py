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

def test_conjugate_gradient() -> None:

    A_1 = A_1 = np.array([[4, 1],
              [1, 3]])

    b_1 = np.array([1, 2])
    n = 5
    expected_1 = np.array([0.09090909, 0.63636364])
    output_1 = conjugate_gradient(A_1, b_1, n)
    assert np.allclose(output_1, expected_1, atol=0.01), f"Test case 1 failed: expected {expected_1}, got {output_1}"

    A_2 = np.array([[4, 1, 2],
                [1, 3, 0],
                [2, 0, 5]])
    b_2 = np.array([7, 8, 5])
    n = 1
    expected_2 = np.array([1.2627451, 1.44313725, 0.90196078])
    output_2 = conjugate_gradient(A_2, b_2, n)
    assert np.allclose(output_2, expected_2, atol=0.01), f"Test case 2 failed: expected {expected_2}, got {output_2}"

    # 5x5 positive definite
    A_3 = np.array([[6, 2, 1, 1, 0],
                [2, 5, 2, 1, 1],
                [1, 2, 6, 1, 2],
                [1, 1, 1, 7, 1],
                [0, 1, 2, 1, 8]])
    b_3 = np.array([1, 2, 3, 4, 5])
    n = 100
    expected_3 = np.array([0.01666667, 0.11666667, 0.21666667, 0.45, 0.5])
    output_3 = conjugate_gradient(A_3, b_3, n)
    assert np.allclose(output_3, expected_3, atol=0.01), f"Test case 3 failed: expected {expected_3}, got {output_3}"


if __name__ == "__main__":
    test_conjugate_gradient()
    print("All Conjugate Gradient tests passed.")