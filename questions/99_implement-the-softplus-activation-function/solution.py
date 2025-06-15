import math


def softplus(x: float) -> float:
    """
    Compute the softplus activation function.

    Args:
        x: Input value

    Returns:
        The softplus value: log(1 + e^x)
    """
    # To prevent overflow for large positive values
    if x > 100:
        return x
    # To prevent underflow for large negative values
    if x < -100:
        return 0.0

    return round(math.log(1.0 + math.exp(x)), 4)
