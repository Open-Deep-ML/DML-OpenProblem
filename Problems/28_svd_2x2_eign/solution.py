import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    y1, x1 = (A[1, 0] + A[0, 1]), (A[0, 0] - A[1, 1])
    y2, x2 = (A[1, 0] - A[0, 1]), (A[0, 0] + A[1, 1])

    h1 = np.sqrt(y1**2 + x1**2)
    h2 = np.sqrt(y2**2 + x2**2)

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

def test_svd_2x2():
    # Test case 1
    A = np.array([[-10, 8], [10, -1]])
    expected_output = (
        np.array([[0.8, -0.6], [-0.6, -0.8]]),
        np.array([15.65247584, 4.47213595]),
        np.array([[-0.89442719, 0.4472136], [-0.4472136, -0.89442719]])
    )
    assert np.allclose(svd_2x2(A)[0], expected_output[0])
    assert np.allclose(svd_2x2(A)[1], expected_output[1])
    assert np.allclose(svd_2x2(A)[2], expected_output[2])

    # Test case 2
    A = np.array([[1, 2], [3, 4]])
    expected_output = (
        np.array([[-0.40455358, -0.9145143], [-0.9145143, 0.40455358]]),
        np.array([5.464, 0.366]),
        np.array([[-0.57604844, -0.81741556], [0.81741556, -0.57604844]])
    )
    assert np.allclose(svd_2x2(A)[0], expected_output[0])
    assert np.allclose(svd_2x2(A)[1], expected_output[1])
    assert np.allclose(svd_2x2(A)[2], expected_output[2])

if __name__ == "__main__":
    test_svd_2x2()
    print("All svd_2x2 tests passed.")
