import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)

    Returns:
        np.ndarray of shape (batch_size, d)
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    d = x.shape[1] // 2
    x1 = x[:, :d]
    x2 = x[:, d:]
    return x1 * (x2 * sigmoid(x2))
