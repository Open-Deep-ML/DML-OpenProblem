import torch


def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    torch.as_tensor(A, dtype=torch.float)
    torch.as_tensor(b, dtype=torch.float)
    # Your implementation here
    pass
