import numpy as np


def phi_transform(data: list[float], degree: int) -> list[list[float]]:
    """
    Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

    Args:
            data (list[float]): A list of numerical values to transform.
            degree (int): The degree of the polynomial expansion.

    Returns:
            list[list[float]]: A nested list where each inner list represents the transformed features of a data point.
    """
    if degree < 0 or not data:
        return []
    return np.array([[x**i for i in range(degree + 1)] for x in data]).tolist()
