import numpy as np

def sparse_window_attention(Q, K, V, window_size, scale_factor=None):
    """
    Computes sparse attention with a sliding window mask to efficiently handle longer context lengths.
    This implementation uses a loop over the sequence to compute attention only within the specified window,
    reducing memory usage compared to dense attention.

    Args:
        Q (np.ndarray): Query matrix of shape (seq_len, d_k)
        K (np.ndarray): Key matrix of shape (seq_len, d_k)
        V (np.ndarray): Value matrix of shape (seq_len, d_v)
        window_size (int): The radius of the attention window (attends to window_size positions on each side).
        scale_factor (float, optional): Scaling factor for the dot product. If None, uses sqrt(d_k).

    Returns:
        np.ndarray: Attention output of shape (seq_len, d_v)
    """
    seq_len = Q.shape[0]
    d_k = Q.shape[1]
    if scale_factor is None:
        scale_factor = np.sqrt(d_k).astype(float)
    output = np.zeros((seq_len, V.shape[1]), dtype=V.dtype)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        local_Q = Q[i:i+1]
        local_K = K[start:end]
        local_V = V[start:end]
        scores = np.dot(local_Q, local_K.T) / scale_factor
        max_score = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_score)
        sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
        attention_weights = exp_scores / sum_exp
        output[i] = np.dot(attention_weights, local_V)
    return output
