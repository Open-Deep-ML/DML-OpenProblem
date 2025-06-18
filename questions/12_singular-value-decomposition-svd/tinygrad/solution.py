import numpy as np
from tinygrad.tensor import Tensor

def svd_2x2_singular_values_tg(A) -> tuple[Tensor, Tensor, Tensor]:
    """
    Approximate the SVD of a 2×2 matrix A using one Jacobi rotation in tinygrad.
    Returns (U, S, Vt) where S is a 1-D Tensor of singular values.
    """
    A_t = Tensor(A).float()
    # compute AᵀA
    a2 = A_t.T.matmul(A_t)
    V = Tensor([[1.0, 0.0], [0.0, 1.0]])
    # extract entries
    a_val = a2[0,0].numpy(); d_val = a2[1,1].numpy(); b_val = a2[0,1].numpy()
    # compute rotation angle
    if np.isclose(a_val, d_val):
        theta = np.pi/4
    else:
        theta = 0.5 * np.arctan2(2 * b_val, a_val - d_val)
    c, s = np.cos(theta), np.sin(theta)
    R = Tensor([[c, -s], [s, c]])
    D = R.T.matmul(a2).matmul(R)
    V = V.matmul(R)
    # singular values
    S = Tensor(np.sqrt([D[0,0].numpy(), D[1,1].numpy()]))
    # compute U
    S_inv = Tensor([[1.0/S[0].numpy(), 0.0], [0.0, 1.0/S[1].numpy()]])
    U = A_t.matmul(V).matmul(S_inv)
    return U, S, V.T
