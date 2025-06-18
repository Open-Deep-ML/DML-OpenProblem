import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    # First weight layer
    y = np.dot(w1, x)
    # First ReLU
    y = np.maximum(0, y)
    # Second weight layer
    y = np.dot(w2, y)
    # Add shortcut connection (x + F(x))
    y = y + x
    # Final ReLU
    y = np.maximum(0, y)
    return y
