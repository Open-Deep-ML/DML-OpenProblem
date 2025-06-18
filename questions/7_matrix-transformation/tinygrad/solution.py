import numpy as np
from tinygrad.tensor import Tensor

def transform_matrix_tg(A, T, S) -> Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S for 2×2 matrices using tinygrad.
    Inputs A, T, S can be Python lists, NumPy arrays, or tinygrad Tensors.
    Returns a 2×2 Tensor or Tensor(-1.) if T or S is singular.
    """
    A_t = Tensor(A).float()
    T_t = Tensor(T).float()
    S_t = Tensor(S).float()
    # manual 2×2 determinant
    detT = T_t[0,0]*T_t[1,1] - T_t[0,1]*T_t[1,0]
    detS = S_t[0,0]*S_t[1,1] - S_t[0,1]*S_t[1,0]
    if detT.numpy() == 0 or detS.numpy() == 0:
        return Tensor(-1.)
    # inverse of 2×2
    a,b,c,d = T_t[0,0], T_t[0,1], T_t[1,0], T_t[1,1]
    T_inv = Tensor([[d, -b], [-c, a]]) / detT
    out = T_inv.matmul(A_t).matmul(S_t)
    # round via NumPy then wrap back
    rounded = np.round(out.numpy(), 3)
    return Tensor(rounded)
