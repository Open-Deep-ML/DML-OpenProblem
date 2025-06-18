import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    d_a = np.diag(A)
    nda = A - np.diag(d_a)
    x = np.zeros(len(b))
    x_hold = np.zeros(len(b))
    for _ in range(n):
        for i in range(len(A)):
            x_hold[i] = (1/d_a[i]) * (b[i] - sum(nda[i]*x))
        x = x_hold.copy()
    return np.round(x, 4).tolist()

def test_solve_jacobi() -> None:
    # Test cases for solve_jacobi function

    # Test case 1
    A = np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]])
    b = np.array([-1, 2, 3])
    assert solve_jacobi(A, b, 2) == [0.146, 0.2032, -0.5175]

    # Test case 2
    A = np.array([[4, 1, 2], [1, 5, 1], [2, 1, 3]])
    b = np.array([4, 6, 7])
    assert solve_jacobi(A, b, 5) == [-0.0806, 0.9324, 2.4422]

    # Test case 3
    A = np.array([[4, 2, -2], [1, -3, -1], [3, -1, 4]])
    b = np.array([0, 7, 5])
    assert solve_jacobi(A, b, 3) == [1.7083, -1.9583, -0.7812]

if __name__ == "__main__":
    test_solve_jacobi()
    print("All solve_jacobi tests passed.")
