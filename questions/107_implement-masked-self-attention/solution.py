import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    d_k = Q.shape[1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    scores = scores + mask  # Apply mask
    attention_weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
    return np.matmul(attention_weights, V)
