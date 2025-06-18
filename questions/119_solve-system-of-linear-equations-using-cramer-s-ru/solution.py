import numpy as np

def cramers_rule(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n, m = A.shape
    if n != m or b.shape[0] != n:
        return -1

    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0):
        return -1

    x = np.zeros(n)
    for i in range(n):
        A_mod = A.copy()
        A_mod[:, i] = b
        det_A_mod = np.linalg.det(A_mod)
        x[i] = det_A_mod / det_A

    return x
