import math

def softplus(x: float, beta: float=1.0, threshold: float=20.0) -> float:
    """
    Implements the Softplus activation function.

    Args:
        x (float): Input Value

    Returns:
        float: The Softplus of the input
    """
    if beta * x > threshold:
        return float(x)
    return round(1 / beta * math.log(1 + math.exp(beta * x)), 4)
