import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.

    Parameters
    ----------
    values : np.ndarray
        Input array (1D or 2D) to be rescaled.
    c : float
        New range lower bound.
    d : float
        New range upper bound.

    Returns
    -------
    np.ndarray
        Scaled array with the same shape as the input.
    """
    a, b = values.min(), values.max()
    return c + (d - c) / (b - a) * (values - a)
