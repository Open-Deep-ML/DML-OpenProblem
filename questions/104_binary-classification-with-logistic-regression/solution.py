import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N × D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    z = np.dot(X, weights) + bias
    z = np.clip(z, -500, 500)  # Prevent overflow in exp
    probabilities = 1 / (1 + np.exp(-z))
    return (probabilities >= 0.5).astype(int)
