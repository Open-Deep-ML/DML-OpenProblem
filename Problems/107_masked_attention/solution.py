import numpy as np
from typing import Tuple

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

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute self-attention for a single head with a mask applied.
    
    Args:
    Q: numpy array of shape (seq_len, d_k), Query matrix
    K: numpy array of shape (seq_len, d_k), Key matrix
    V: numpy array of shape (seq_len, d_k), Value matrix
    mask: numpy array of shape (seq_len, seq_len), Mask matrix
    
    Returns:
    attention_output: numpy array of shape (seq_len, d_k), output of the masked-attention mechanism
    """
    d_k = Q.shape[1]  # Get the dimension of the keys
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)  # Compute scaled dot-product attention scores
    
    # Apply the mask by adding a large negative value to the masked positions
    scores = scores + mask  # This will set the masked positions to a large negative value (-inf)
    
    # For numerical stability, compute softmax
    attention_weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # Subtract max for numerical stability
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)  # Normalize

    # Compute the final attention output
    attention_output = np.matmul(attention_weights, V)
    return attention_output


def test_masked_attention():
    # Test case 1: Basic functionality with computed Q, K, V
    m, n = 6, 8
    np.random.seed(42)
    X = np.arange(m*n).reshape(m,n)
    X = np.random.permutation(X.flatten()).reshape(m, n)
    mask = np.triu(np.ones((m, m))*(-np.inf), k=1)
    W_q = np.random.randint(0,4,size=(n,n))
    W_k = np.random.randint(0,5,size=(n,n))
    W_v = np.random.randint(0,6,size=(n,n))
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)

    # test masked attention
    actual_output = masked_attention(Q, K, V, mask)
    expected_output = np.array([[547., 490., 399., 495., 485., 439., 645., 393.],
                                [547., 490., 399., 495., 485., 439., 645., 393.],
                                [471., 472., 429., 538., 377., 450., 531., 362.],
                                [471., 472., 429., 538., 377., 450., 531., 362.],
                                [471., 472., 429., 538., 377., 450., 531., 362.],
                                [471., 472., 429., 538., 377., 450., 531., 362.]])
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 1 failed")

    # test different shape
    m, n = 4, 4
    np.random.seed(42)
    X = np.arange(m*n).reshape(m,n)
    X = np.random.permutation(X.flatten()).reshape(m, n)
    mask = np.triu(np.ones((m, m))*(-np.inf), k=1)
    W_q = np.random.randint(0,4,size=(n,n))
    W_k = np.random.randint(0,5,size=(n,n))
    W_v = np.random.randint(0,6,size=(n,n))
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    actual_output = masked_attention(Q, K, V, mask)
    expected_output = np.array([[ 52.,  63.,  48.,  71.],
                                [103., 109.,  46.,  99.],
                                [103., 109.,  46.,  99.],
                                [103., 109.,  46.,  99.]])
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 2 failed")

if __name__ == "__main__":
    test_masked_attention()
    print("All masked-attention tests passed.")