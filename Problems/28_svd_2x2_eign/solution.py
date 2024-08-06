import numpy as np


def svd_2x2(A: np.ndarray) -> tuple:
    y1, x1 = (A[1, 0] + A[0, 1]), (A[0, 0] - A[1, 1])
    y2, x2 = (A[1, 0] - A[0, 1]), (A[0, 0] + A[1, 1])

    h1 = np.sqrt(y1 ** 2 + x1 ** 2)
    h2 = np.sqrt(y2 ** 2 + x2 ** 2)

    t1 = x1 / h1
    t2 = x2 / h2

    cc = np.sqrt((1.0 + t1) * (1.0 + t2))
    ss = np.sqrt((1.0 - t1) * (1.0 - t2))
    cs = np.sqrt((1.0 + t1) * (1.0 - t2))
    sc = np.sqrt((1.0 - t1) * (1.0 + t2))

    c1, s1 = (cc - ss) / 2.0, (sc + cs) / 2.0
    U = np.array([[-c1, -s1], [-s1, c1]])

    s = np.array([(h1 + h2) / 2.0, abs(h1 - h2) / 2.0])

    V = np.diag(1.0 / s) @ U.T @ A

    return U, s, V


def check_svd(U, s, V, A):
    def is_equal(A: np.ndarray, B: np.ndarray, precision=1e-10) -> bool:
        return (np.abs(A - B) < precision).all()

    # SVD Condition:
    # 1. U*S*V.T == A
    # 2. U and V are orthogonal matrix -> U*U.T == E, V*V.T == E
    result = U @ np.diag(s) @ V
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
    test_svd_2x2()
    print("All svd_2x2 tests passed.")
