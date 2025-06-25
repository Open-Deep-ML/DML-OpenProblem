import numpy as np

def batch_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Batch Normalization.
    
    Args:
    X: numpy array of shape (B, C, H, W), input data
    gamma: numpy array of shape (C,), scale parameter
    beta: numpy array of shape (C,), shift parameter
    epsilon: small constant to avoid division by zero
    
    Returns:
    norm_X: numpy array of shape (B, C, H, W), normalized output
    """
    # Compute mean and variance across the batch and spatial dimensions
    mean = np.mean(X, axis=(0, 2, 3), keepdims=True)  # Mean over (B, H, W)
    variance = np.var(X, axis=(0, 2, 3), keepdims=True)  # Variance over (B, H, W)
    
    # Normalize
    X_norm = (X - mean) / np.sqrt(variance + epsilon)
    
    # Scale and shift
    norm_X = gamma * X_norm + beta
    return norm_X

# Test cases for batch normalization
def test_batch_normalizations():
    # Test case: Batch Normalization
    B, C, H, W = 2, 2, 2, 2  # Batch size, Channels, Height, Width
    np.random.seed(42)
    X = np.random.randn(B, C, H, W)
    gamma = np.ones(C).reshape(1, C, 1, 1)
    beta = np.zeros(C).reshape(1, C, 1, 1)

    # Test batch normalization
    actual_output = batch_normalization(X, gamma, beta)
    expected_output = [[[[ 0.42859934, -0.51776438],
                         [ 0.65360963,  1.95820707]],
                        [[ 0.02353721,  0.02355215],
                         [ 1.67355207,  0.93490043]]],
                       [[[-1.01139563,  0.49692747],
                        [-1.00236882, -1.00581468]],
                        [[ 0.45676349, -1.50433085],
                        [-1.33293647, -0.27503802]]]]
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 1 failed")

    # Test different input
    np.random.seed(101)
    X = np.random.randn(B, C, H, W)
    actual_output = batch_normalization(X, gamma, beta)
    expected_output = [[[[ 1.81773164,  0.16104096],
                         [ 0.38406453,  0.06197112]],
                        [[ 1.00432932 ,-0.37139956],
                         [-1.12098938,  0.94031919]]],
                       [[[-1.94800122,  0.25029395],
                         [ 0.08188579, -0.80898678]],
                        [[ 0.34878049, -0.99452891],
                         [-1.24171594,  1.43520478]]]]
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 2 failed")

    # Test different params
    gamma = np.ones(C).reshape(1, C, 1, 1) * 0.5
    beta = np.ones(C).reshape(1, C, 1, 1)
    actual_output = batch_normalization(X, gamma, beta)
    expected_output = [[[[1.90886582, 1.08052048],
                         [1.19203227, 1.03098556]],
                        [[1.50216466, 0.81430022],
                         [0.43950531, 1.4701596 ]]],
                       [[[0.02599939, 1.12514697],
                         [1.04094289, 0.59550661]],
                        [[1.17439025, 0.50273554],
                         [0.37914203, 1.71760239]]]]
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 3 failed")

if __name__ == "__main__":
    test_batch_normalizations()
    print("All normalization tests passed.")