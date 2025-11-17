import numpy as np


def compute_qkv(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute query, key and value matrices from input embeddings (of length dim_in).

    x: (n_tokens, dim_in) input embeddings
    W_q: (dim_in, dim_qk) query weights
    W_k: (dim_in, dim_qk) key weights
    W_v: (dim_in, dim_v) value weights
    Returns (Q, K, V) with shapes (n_tokens, dim_qk), (n_tokens, dim_qk), (n_tokens, dim_v)
    """
    # TODO: return (Q, K, V)
    pass


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Apply softmax along the given axis.

    x: input array
    axis: the axis to normalize along
    Returns array of same shape where values along `axis` sum to 1
    """
    # TODO: return softmax_output
    pass


def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot product self attention.

    Q: (n_tokens, dim_qk) queries
    K: (n_tokens, dim_qk) keys
    V: (n_tokens, dim_v) values
    Returns attention output of shape (n_tokens, dim_v)
    """
    # TODO: return attention_output
    pass




