import numpy as np
from tinygrad.tensor import Tensor

def solve_jacobi_tg(A, b, n) -> Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations in tinygrad.
    A: list of lists or Tensor; b: list or Tensor; n: number of iterations.
    Returns a 1-D Tensor of length m, rounded to 4 decimals.
    """
    A_t = Tensor(A).float()
    b_t = Tensor(b).float()
    m = A_t.shape[0]
    # extract diagonal
    d_list = [A_t[i,i] for i in range(m)]
    d = Tensor(d_list)
    # build remainder matrix
    nda_list = [[A_t[i,j] if i != j else 0 for j in range(m)] for i in range(m)]
    nda = Tensor(nda_list).float()
    x = Tensor([0.0]*m).float()
    for _ in range(n):
        x = (b_t - nda.matmul(x)) / d
    res = x.numpy()
    return Tensor(np.round(res, 4))
