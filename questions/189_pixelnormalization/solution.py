import numpy as np

def pixel_normalization(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return X / np.sqrt(np.mean(X**2, axis=1, keepdims=True) + eps)