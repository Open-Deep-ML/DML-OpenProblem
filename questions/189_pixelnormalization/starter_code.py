import numpy as np

def pixel_normalization(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Perform pixel normalization on the input array X.
    Each pixel value is divided by the square root of the mean of the squared pixel values
    across each row, plus a small epsilon for numerical stability."""
    # Your code here
    pass