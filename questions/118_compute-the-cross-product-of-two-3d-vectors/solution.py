import numpy as np


def cross_product(a, b):
    """
    Compute the cross product of two 3D vectors a and b.
    Parameters:
        a (array-like): A 3-element vector.
        b (array-like): A 3-element vector.
    Returns:
        numpy.ndarray: The cross product vector.
    """
    a = np.array(a)
    b = np.array(b)

    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("Both input vectors must be of length 3.")

    cross = np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )
    return cross
