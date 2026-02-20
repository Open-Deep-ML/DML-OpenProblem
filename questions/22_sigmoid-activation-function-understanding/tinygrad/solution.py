from tinygrad.tensor import Tensor


def sigmoid_tg(z: float) -> float:
    """
    Compute the sigmoid activation function using tinygrad.
    Input:
      - z: float or tinygrad Tensor scalar
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    t = Tensor(z)
    res = (Tensor(1.0) / (Tensor(1.0) + (-t).exp())).numpy().tolist()
    return round(res, 4)
