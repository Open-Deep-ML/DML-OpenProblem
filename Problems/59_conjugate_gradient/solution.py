import numpy as np

def conjugate_gradient(A: np.array, b: np.array, x0: np.array, n: int, tol: float) -> np.array:
    pass

def residual(A: np.array, b: np.array, x: np.array) -> np.array:
    # calculate linear system residuals
    return b - A @ x

def alpha(A: np.array, r: np.array, p: np.array) -> float:
    
    pA_den = p @ A
    alpha_num = np.dot(r, r)
    alpha_den = np.dot(pA_den, p)

    return alpha_num/alpha_den

def beta(r: np.array, r_plus1: np.array) -> float:
    
    beta_num = np.dot(r_plus1, r_plus1)
    beta_den = np.dot(r, r)

    return beta_num/beta_den

def test_conjugate_gradient() -> None:
    pass

if __name__ == "__main__":
    test_conjugate_gradient()
    print("All tests passed.")