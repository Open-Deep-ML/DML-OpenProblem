def softsign(x: float) -> float:
    """
    Implements the Softsign activation function.

    Args:
        x (float): Input value

    Returns:
        float: The Softsign of the input, calculated as x/(1 + |x|)
    """
    return round(x / (1 + abs(x)), 4)
