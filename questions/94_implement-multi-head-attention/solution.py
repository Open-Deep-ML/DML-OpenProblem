import numpy as np
from typing import Tuple


def compute_qkv(
    X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Query (Q), Key (K), and Value (V) matrices.

    Args:
    X: numpy array of shape (seq_len, d_model), input sequence
    W_q, W_k, W_v: numpy arrays of shape (d_model, d_model), weight matrices for Q, K, and V

    Returns:
    Q, K, V: numpy arrays of shape (seq_len, d_model)
    """
    Q = np.dot(X, W_q)  # Compute the Query matrix Q
    K = np.dot(X, W_k)  # Compute the Key matrix K
    V = np.dot(X, W_v)  # Compute the Value matrix V
    return Q, K, V


def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute self-attention for a single head.

    Args:
    Q: numpy array of shape (seq_len, d_k), Query matrix
    K: numpy array of shape (seq_len, d_k), Key matrix
    V: numpy array of shape (seq_len, d_k), Value matrix

    Returns:
    attention_output: numpy array of shape (seq_len, d_k), output of the self-attention mechanism
    """
    d_k = Q.shape[1]  # Get the dimension of the keys
    scores = np.matmul(Q, K.T) / np.sqrt(
        d_k
    )  # Compute scaled dot-product attention scores
    score_max = np.max(
        scores, axis=1, keepdims=True
    )  # Find the maximum score for numerical stability
    attention_weights = np.exp(scores - score_max) / np.sum(
        np.exp(scores - score_max), axis=1, keepdims=True
    )  # Compute softmax to get attention weights
    attention_output = np.matmul(
        attention_weights, V
    )  # Compute the final attention output
    return attention_output


def multi_head_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int
) -> np.ndarray:
    """
    Compute multi-head attention.

    Args:
    Q, K, V: numpy arrays of shape (seq_len, d_model), Query, Key, and Value matrices
    n_heads: int, number of attention heads

    Returns:
    attention_output: numpy array of shape (seq_len, d_model), final attention output
    """
    d_model = Q.shape[1]  # Get the model dimension
    assert d_model % n_heads == 0  # Ensure d_model is divisible by n_heads
    d_k = d_model // n_heads  # Dimension for each head

    # Reshape Q, K, V to separate heads
    Q_reshaped = Q.reshape(Q.shape[0], n_heads, d_k).transpose(
        1, 0, 2
    )  # Reshape and transpose to (n_heads, seq_len, d_k)
    K_reshaped = K.reshape(K.shape[0], n_heads, d_k).transpose(
        1, 0, 2
    )  # Reshape and transpose to (n_heads, seq_len, d_k)
    V_reshaped = V.reshape(V.shape[0], n_heads, d_k).transpose(
        1, 0, 2
    )  # Reshape and transpose to (n_heads, seq_len, d_k)

    # Compute attention scores for each head
    attentions = []  # Store attention outputs for each head

    for i in range(n_heads):
        attn = self_attention(
            Q_reshaped[i], K_reshaped[i], V_reshaped[i]
        )  # Compute attention for the i-th head
        attentions.append(attn)  # Collect attention output

    # Concatenate all head outputs
    attention_output = np.concatenate(
        attentions, axis=-1
    )  # Concatenate along the last axis (columns)
    return attention_output  # Return the final attention output
