import numpy as np


def svd_2x2(A: np.ndarray) -> tuple:
    AAt = np.dot(A, A.transpose())

    eig_values, U = np.linalg.eig(AAt)
    if eig_values[0] < eig_values[1]:
        U[:, 0] += U[:, 1]
        U[:, 1] = U[:, 0] - U[:, 1]
        U[:, 0] -= U[:, 1]
        eig_values[0] += eig_values[1]
        eig_values[1] = eig_values[0] - eig_values[1]
        eig_values[0] -= eig_values[1]

    singular_values = np.sqrt(eig_values)
    V = np.dot(np.dot(A.transpose(), U), np.diag(1 / singular_values))

    return U, singular_values, V


def check_svd(U, s, V, A):
    def is_equal(A: np.ndarray, B: np.ndarray, precision=1e-10) -> bool:
        return (np.abs(A - B) < precision).all()

    # SVD Condition:
    # 1. U*S*V.T == A
    # 2. U and V are orthogonal matrix -> U*U.T == E, V*V.T == E
    result = U @ np.diag(s) @ V.T
    return is_equal(result, A) and is_equal(U @ U.T, np.eye(U.shape[0])) and is_equal(V @ V.T, np.eye(V.shape[0]))


def test_svd_2x2():
    # Test case 1
    A = np.array([[-10, 8], [10, -1]])

    U, S, V = svd_2x2(A)
    assert check_svd(U, S, V, A)

    # Test case 2
    A = np.array([[1, 2], [3, 4]])

    U, S, V = svd_2x2(A)
    assert check_svd(U, S, V, A)


if __name__ == "__main__":
    A = np.array([[-10, 8], [10, -1]])

    test_svd_2x2()
    print("All svd_2x2 tests passed.")
