import numpy as np

def sinkhorn_knopp(H_res: np.ndarray, iterations: int = 20) -> np.ndarray:
    """
    Apply the Sinkhorn–Knopp algorithm to the mixing matrix

    Parameters: 
        H_res: A 2D (NxN) unconstrained residual mixing matrix.
        iterations: Number of Sinkhorn normalization iterations to perform.

    Returns
        Doubly stochastic matrix in the Birkhoff Polytope
    """
    # your code here
    pass