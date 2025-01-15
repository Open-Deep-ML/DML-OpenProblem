import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    # vectorized solution (づ｡◕‿‿◕｡)づ
    x = np.zeros_like(b)
    A_diag_vec = np.diag(A)
    for epoch in range(n):
        x = 1/A_diag_vec*(b-(A@x-A_diag_vec*x))
    return x

import numpy as np

def test_solve_jacobi() -> None:
    def lists_are_close(list1, list2, tol=1e-4):
        """Helper function to check if two lists are element-wise close."""
        return np.allclose(list1, list2, atol=tol)

    # Test case 1
    A = np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]])
    b = np.array([-1, 2, 3])
    result = solve_jacobi(A, b, 2)
    expected = [0.146, 0.2032, -0.5175]
    assert lists_are_close(result, expected), f"Test case 1 failed: {result} != {expected}"

    # Test case 2
    A = np.array([[4, 1, 2], [1, 5, 1], [2, 1, 3]])
    b = np.array([4, 6, 7])
    result = solve_jacobi(A, b, 5)
    expected = [-0.0806, 0.9324, 2.4422]
    assert lists_are_close(result, expected), f"Test case 2 failed: {result} != {expected}"

    # Test case 3
    A = np.array([[4, 2, -2], [1, -3, -1], [3, -1, 4]])
    b = np.array([0, 7, 5])
    result = solve_jacobi(A, b, 3)
    expected = [1.7083, -1.9583, -0.7812]
    assert lists_are_close(result, expected), f"Test case 3 failed: {result} != {expected}"

    print("All test cases passed!")



if __name__ == "__main__":
    test_solve_jacobi()
    print("All solve_jacobi tests passed.")