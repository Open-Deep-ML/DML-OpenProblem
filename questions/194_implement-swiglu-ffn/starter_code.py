import numpy as np

# Implement your function below.
def swiglu_ffn(x, W1, W3, W2):
    """
    Implement the SwiGLU Feed-Forward Network.

    Args:
        x (np.ndarray): Input array of shape (batch_size, hidden_dim).
        W1 (np.ndarray): Gate projection weight of shape (intermediate_dim, hidden_dim).
        W3 (np.ndarray): Up projection weight of shape (intermediate_dim, hidden_dim).
        W2 (np.ndarray): Down projection weight of shape (hidden_dim, intermediate_dim).

    Returns:
        np.ndarray: Output of shape (batch_size, hidden_dim).
    """
    pass
