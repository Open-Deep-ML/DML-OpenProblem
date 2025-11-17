import numpy as np


def compute_qkv(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = np.dot(x, W_q)
    K = np.dot(x, W_k)
    V = np.dot(x, W_v)
    return Q, K, V


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    d_k = K.shape[1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=1)
    attention_output = np.matmul(attention_weights, V)
    return attention_output
