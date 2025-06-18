import torch

def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    d = torch.diag(A_t)
    nda = A_t - torch.diag(d)
    x = torch.zeros_like(b_t)
    for _ in range(n):
        x = (b_t - nda.matmul(x)) / d
    return torch.round(x * 10000) / 10000
