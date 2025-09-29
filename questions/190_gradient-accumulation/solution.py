import numpy as np

def accumulate_gradients(grad_list):
    """
    Accumulates (sums) a list of gradient arrays into a single array.

    Args:
        grad_list (list of np.ndarray): List of gradient arrays, all of the same shape.

    Returns:
        np.ndarray: The accumulated (summed) gradients, same shape as input arrays.
    """
    return np.sum(grad_list, axis=0).astype(float)
