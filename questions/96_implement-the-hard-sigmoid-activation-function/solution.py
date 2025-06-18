def hard_sigmoid(x: float) -> float:
    """
    Implements the Hard Sigmoid activation function.

    Args:
        x (float): Input value

    Returns:
        float: The Hard Sigmoid of the input
    """
    if x <= -2.5:
        return 0.0
    elif x >= 2.5:
        return 1.0
    else:
        return 0.2 * x + 0.5
