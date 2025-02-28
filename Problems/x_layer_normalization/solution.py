import numpy as np
from typing import Tuple

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Layer Normalization.
    
    Args:
    X: numpy array of shape (batch_size, seq_len, d_model), input data
    gamma: numpy array of shape (d_model,), scale parameter
    beta: numpy array of shape (d_model,), shift parameter
    epsilon: small constant to avoid division by zero
    
    Returns:
    norm_X: numpy array of shape (batch_size, seq_len, d_model), normalized output
    """
    # Compute mean and variance across the last dimension (d_model)
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)
    
    # Normalize the input
    X_norm = (X - mean) / np.sqrt(variance + epsilon)
    
    # Scale and shift
    norm_X = gamma * X_norm + beta
    return norm_X

# Test cases for each normalization
def test_normalizations():
    # Test case: Instance Normalization
    batch_size, seq_len, d_model = 2, 2, 3  # batch_size, seq_len, d_model
    np.random.seed(42)
    X = np.random.randn(batch_size, seq_len, d_model)
    gamma = np.ones(d_model).reshape(1, 1, -1)
    beta = np.zeros(d_model).reshape(1, 1, -1)

    actual_output = layer_normalization(X, gamma, beta)
    expected_output = [[[ 0.47373971, -1.39079736,  0.91705765],
                        [ 1.41420326, -0.70711154, -0.70709172]],
                       [[ 1.13192477,  0.16823009, -1.30015486],
                        [ 1.4141794,  -0.70465482, -0.70952458]]]
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 1 failed")

    # Test different input
    batch_size, seq_len, d_model = 2, 3, 4 # batch_size, seq_len, d_model
    X = np.random.randn(batch_size, seq_len, d_model)
    gamma = np.ones(d_model).reshape(1, 1, -1)
    beta = np.zeros(d_model).reshape(1, 1, -1)
    actual_output = layer_normalization(X, gamma, beta)
    expected_output = [[[ 1.40051929, -1.05033782, -0.83613949,  0.48595802],
                        [-0.40001933,  1.65674313, -0.23758493, -1.01913888],
                        [ 1.45374525, -0.19102048,  0.09419298, -1.35691775]],
                       [[-0.4080207,   0.69596352, -1.42997001,  1.1420272 ],
                        [-0.67302065, -0.3717648,  -0.67406199,  1.71884744],
                        [ 0.4270937,  -0.83315759,  1.43610477, -1.03004088]]]
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 2 failed")

    # Test different params
    gamma = np.ones(d_model).reshape(1, 1, -1) * 0.5
    beta = np.ones(d_model).reshape(1, 1, -1)
    actual_output = layer_normalization(X, gamma, beta)
    expected_output = [[[1.70025965, 0.47483109, 0.58193026, 1.24297901],
                        [0.79999034, 1.82837157, 0.88120754, 0.49043056],
                        [1.72687263, 0.90448976, 1.04709649, 0.32154112]],
                       [[0.79598965, 1.34798176, 0.28501499, 1.5710136 ],
                        [0.66348968, 0.8141176,  0.662969,   1.85942372],
                        [1.21354685, 0.5834212,  1.71805239, 0.48497956]]]
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 3 failed")
    

if __name__ == "__main__":
    test_normalizations()
    print("All normalization tests passed.")