import numpy as np


def batch_normalization(
    X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    # Compute mean and variance across the batch and spatial dimensions
    mean = np.mean(X, axis=(0, 2, 3), keepdims=True)  # Mean over (B, H, W)
    variance = np.var(X, axis=(0, 2, 3), keepdims=True)  # Variance over (B, H, W)
    # Normalize
    X_norm = (X - mean) / np.sqrt(variance + epsilon)
    # Scale and shift
    norm_X = gamma * X_norm + beta
    return norm_X
