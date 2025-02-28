import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Group Normalization.
    
    Args:
    X: numpy array of shape (B, C, H, W), input data
    gamma: numpy array of shape (C,), scale parameter
    beta: numpy array of shape (C,), shift parameter
    num_groups: number of groups for normalization
    epsilon: small constant to avoid division by zero
    
    Returns:
    norm_X: numpy array of shape (B, C, H, W), normalized output
    """
    batch_size, num_channels, height, width = X.shape
    group_size = num_channels // num_groups

    # Reshape X into groups
    X_reshaped = X.reshape(batch_size, num_groups, group_size, height, width)
    
    # Compute mean and variance for each group
    mean = np.mean(X_reshaped, axis=(2, 3, 4), keepdims=True)
    variance = np.var(X_reshaped, axis=(2, 3, 4), keepdims=True)
    
    X_norm = (X_reshaped - mean) / np.sqrt(variance + epsilon)
    
    # Reshape back to the original shape
    X_norm = X_norm.reshape(batch_size, num_channels, height, width)
    
    # Apply scale and shift
    norm_X = gamma * X_norm + beta
    return norm_X

# Test cases for each normalization
def test_normalizations():
    # Test case: Group Normalization
    B, C, H, W = 2, 2, 2, 2  # Batch size, Channels, Height, Width
    np.random.seed(42)
    X = np.random.randn(B, C, H, W)
    gamma = np.ones(C).reshape(1, C, 1, 1)
    beta = np.zeros(C).reshape(1, C, 1, 1)
    
    num_groups = 2
    actual_output = group_normalization(X, gamma, beta, num_groups)
    expected_output = [[[[-0.22869287, -1.29977477],
                         [ 0.02597078,  1.50249686]],
                        [[-0.92595704, -0.92593544],
                         [ 1.45999914,  0.39189334]]],
                       [[[-0.58480728,  1.73198422],
                         [-0.57094204, -0.57623491]],
                        [[ 1.40051929, -1.05033782],
                         [-0.83613949,  0.48595802]]]]
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 1 failed")

    # Test different input
    np.random.seed(101)
    X = np.random.randn(B, C, H, W)
    actual_output = group_normalization(X, gamma, beta, num_groups)
    expected_output = [[[[ 1.70844399, -0.62774597],
                         [-0.31324826, -0.76744976]],
                        [[ 0.99084778, -0.53859496],
                         [-1.37193845,  0.91968563]]],
                       [[[-1.53697108,  0.98107798],
                         [ 0.78817395, -0.23228085]],
                        [[ 0.4278294,  -0.81654215],
                         [-1.04552328,  1.43423603]]]]
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 2 failed")

    # Test different params
    gamma = np.ones(C).reshape(1, C, 1, 1) * 0.5
    beta = np.ones(C).reshape(1, C, 1, 1)
    actual_output = group_normalization(X, gamma, beta, num_groups)
    expected_output = [[[[1.854222,   0.68612701],
                         [0.84337587, 0.61627512]],
                        [[1.49542389, 0.73070252],
                         [0.31403077, 1.45984282]]],
                       [[[0.23151446, 1.49053899],
                         [1.39408697, 0.88385958]],
                        [[1.2139147,  0.59172892],
                         [0.47723836, 1.71711802]]]]
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 3 failed")

if __name__ == "__main__":
    test_normalizations()
    print("All normalization tests passed.")