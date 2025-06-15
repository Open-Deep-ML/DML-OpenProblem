from tinygrad.tensor import Tensor


def calculate_eigenvalues_tg(matrix) -> Tensor:
    """
    Compute eigenvalues of a 2×2 matrix using tinygrad.
    Input: 2×2 list, NumPy array, or Tensor; Output: 1-D Tensor with eigenvalues in ascending order.
    """
    m = Tensor(matrix).float()
    a = m[0, 0]
    b = m[0, 1]
    c = m[1, 0]
    d = m[1, 1]
    trace = a + d
    det = a * d - b * c
    disc = trace * trace - 4 * det
    sqrt_disc = disc.pow(0.5)
    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2
    vals = sorted([lambda1.numpy(), lambda2.numpy()])
    return Tensor(vals)
