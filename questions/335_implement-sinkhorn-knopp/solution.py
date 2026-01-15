import numpy as np

def sinkhorn_knopp(H_res: np.ndarray, iterations: int = 20) -> np.ndarray:
    """
    Apply the Sinkhorn–Knopp algorithm to the mixing matrix

    Parameters: 
        H_res: A non-negative 2D matrix unconstrained residual mixing matrix.
        iterations: Number of Sinkhorn normalization iterations to perform.

    Returns
        Doubly stochastic matrix in the Birkhoff Polytope
    """
    # Make all elements positive
    H = np.exp(H_res)

    # Sinkhorn-Knopp Iteration  
    for _ in range(iterations):
        # Normalize Columns
        H = H / H.sum(axis=0, keepdims=True)
        # Normalize rows
        H = H / H.sum(axis=1, keepdims=True)
    return H