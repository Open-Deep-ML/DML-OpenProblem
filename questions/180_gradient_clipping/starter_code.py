import numpy as np

# Implement your function below.
def clip_gradients(gradients, max_norm):
    """
    Clips the gradients so that their L2 norm does not exceed max_norm.
    If the L2 norm is less than or equal to max_norm, returns the gradients unchanged.
    Otherwise, scales the gradients so that their L2 norm equals max_norm.

    Args:
        gradients (np.ndarray): The input gradients (any shape).
        max_norm (float): The maximum allowed L2 norm.

    Returns:
        np.ndarray: The clipped gradients, same shape as input.
    """
    pass
