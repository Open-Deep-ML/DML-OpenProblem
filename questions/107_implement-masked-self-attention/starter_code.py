import numpy as np


def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)


def masked_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    # Your code here
    pass
