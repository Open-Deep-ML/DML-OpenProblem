import numpy as np

def self_attention(query, key, value):
    """
    Compute self-attention.
    
    Args:
    query, key, value: numpy arrays of shape (seq_len, d_model)
    
    Returns:
    attention_output: numpy array of shape (seq_len, d_model)
    """
    d_k = query.shape[1]
    
    # Compute attention scores
    scores = np.matmul(query, key.T) / np.sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Compute attention output
    attention_output = np.matmul(attention_weights, value)
    
    return attention_output

def test_self_attention():
    # Test case 1: Basic functionality
    query = np.array([[1, 0], [0, 1]])
    key = np.array([[1, 0], [0, 1]])
    value = np.array([[1, 2], [3, 4]])
    
    expected_output = np.array([[1.660477, 2.660477], [2.339523, 3.339523]])
    actual_output = self_attention(query, key, value)
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, 
                                         err_msg="Test case 1 failed")

    # Test case 2: Different query, key, and value
    query = np.array([[1, 1], [1, 0]])
    key = np.array([[1, 0], [0, 1]])
    value = np.array([[1, 2], [3, 4]])
    
    expected_output = np.array([[2, 3], [1.660477, 2.660477]])
    actual_output = self_attention(query, key, value)
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, 
                                         err_msg="Test case 2 failed")

    # Test case 3: Larger input
    query = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    key = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    expected_output = np.array([[4.61987385, 5.61987385, 6.61987385],
                                [4, 5, 6],
                                [3.38012615, 4.38012615 ,5.38012615]])
    actual_output = self_attention(query, key, value)
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, 
                                         err_msg="Test case 3 failed")

if __name__ == "__main__":
    test_self_attention()
    print("All self-attention tests passed.")