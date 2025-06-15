import numpy as np
from typing import Tuple, List

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)  # Compute scaled dot-product attention scores
    score_max = np.max(scores, axis=1, keepdims=True)  # Find the maximum score for numerical stability
    attention_weights = np.exp(scores - score_max) / np.sum(np.exp(scores - score_max), axis=1, keepdims=True)  # Compute softmax to get attention weights
    attention_output = np.matmul(attention_weights, V)  # Compute the final attention output
    return attention_output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> Tuple[np.ndarray, List[np.ndarray]]:
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
    Q_reshaped = Q.reshape(Q.shape[0], n_heads, d_k).transpose(1, 0, 2)  # Reshape and transpose to (n_heads, seq_len, d_k)
    K_reshaped = K.reshape(K.shape[0], n_heads, d_k).transpose(1, 0, 2)  # Reshape and transpose to (n_heads, seq_len, d_k)
    V_reshaped = V.reshape(V.shape[0], n_heads, d_k).transpose(1, 0, 2)  # Reshape and transpose to (n_heads, seq_len, d_k)

    # Compute attention scores for each head
    attentions = []  # Store attention outputs for each head

    for i in range(n_heads):
        attn = self_attention(Q_reshaped[i], K_reshaped[i], V_reshaped[i])  # Compute attention for the i-th head
        attentions.append(attn)  # Collect attention output
    
    # Concatenate all head outputs
    attention_output = np.concatenate(attentions, axis=-1)  # Concatenate along the last axis (columns)
    return attention_output  # Return the final attention output

def test_multi_head_attention():
    # Test case 1: Basic functionality with computed Q, K, V
    m, n = 6, 8
    n_heads = 4
    np.random.seed(42)
    X = np.arange(m*n).reshape(m,n)
    X = np.random.permutation(X.flatten()).reshape(m, n)
    W_q = np.random.randint(0,4,size=(n,n))
    W_k = np.random.randint(0,5,size=(n,n))
    W_v = np.random.randint(0,6,size=(n,n))
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)

    # test multi-head attention
    actual_output = multi_head_attention(Q, K, V, n_heads)
    expected_output = np.array([[500, 463, 399, 495, 377, 450, 531, 362],
                                [500, 463, 399, 495, 377, 450, 531, 362],
                                [500, 463, 399, 495, 377, 450, 531, 362],
                                [500, 463, 399, 495, 377, 450, 531, 362],
                                [500, 463, 399, 495, 377, 450, 531, 362],
                                [500, 463, 399, 495, 377, 450, 531, 362]])
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 1 failed")

    # test less number of heads
    n_heads = 2
    actual_output = multi_head_attention(Q, K, V, n_heads)
    expected_output = np.array([[547, 490, 399, 495, 377, 450, 531, 362],
                                [547, 490, 399, 495, 377, 450, 531, 362],
                                [547, 490, 399, 495, 377, 450, 531, 362],
                                [547, 490, 399, 495, 377, 450, 531, 362],
                                [547, 490, 399, 495, 377, 450, 531, 362],
                                [547, 490, 399, 495, 377, 450, 531, 362]])
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 2 failed")

    # test small size input
    m, n = 4, 4
    n_heads = 2
    np.random.seed(42)
    X = np.arange(m*n).reshape(m,n)
    X = np.random.permutation(X.flatten()).reshape(m, n)
    W_q = np.random.randint(0,4,size=(n,n))
    W_k = np.random.randint(0,5,size=(n,n))
    W_v = np.random.randint(0,6,size=(n,n))
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    actual_output = multi_head_attention(Q, K, V, n_heads)
    expected_output = np.array([[103, 109, 46, 99],
                                [103, 109, 46, 99],
                                [103, 109, 46, 99],
                                [103, 109, 46, 99]])
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 3 failed")

if __name__ == "__main__":
    test_multi_head_attention()
    print("All multi-head-attention tests passed.")