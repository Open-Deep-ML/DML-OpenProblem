import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    basis = []
    for v in vectors:
        v = np.array(v, dtype=float)
        for b in basis:
            v = v - np.dot(v, b) * b
        norm = np.sqrt(np.dot(v, v))
        if norm > tol:
            v = v / norm
            basis.append(v)
    return basis
