import numpy as np

def instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Instance Normalization.
    
    Args:
    X: numpy array of shape (B, C, H, W), input data
    gamma: numpy array of shape (C,), scale parameter
    beta: numpy array of shape (C,), shift parameter
    epsilon: small constant to avoid division by zero
    
    Returns:
    norm_X: numpy array of shape (B, C, H, W), normalized output
    """
    # Compute mean and variance across the feature dimensions
    mean = np.mean(X, axis=(2, 3), keepdims=True)  # Mean over (H, W)
    variance = np.var(X, axis=(2, 3), keepdims=True)  # Variance over (H, W)
    
    # Normalize
    X_norm = (X - mean) / np.sqrt(variance + epsilon)
    
    # Scale and shift
    norm_X = gamma * X_norm + beta
    return norm_X


# Test cases for each normalization
def test_normalizations():
    # Test case: Instance Normalization
    B, C, H, W = 2, 2, 2, 2  # Batch size, Channels, Height, Width
    np.random.seed(42)
    X = np.random.randn(B, C, H, W)
    gamma = np.ones(C).reshape(1, C, 1, 1)
    beta = np.zeros(C).reshape(1, C, 1, 1)
    
    num_groups = 2
    actual_output = instance_normalization(X, gamma, beta, num_groups)
    expected_output = [[[[-0.08841405, -0.50250083],
                         [ 0.01004046,  0.58087442]],
                        [[-0.43833369, -0.43832346],
                         [ 0.69114093,  0.18551622]]],
                       [[[-0.17259136,  0.51115219],
                         [-0.16849938, -0.17006144]],
                        [[ 0.73955155, -0.55463639],
                         [-0.44152783,  0.25661268]]]]
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 1 failed")

    # Test different input
    np.random.seed(101)
    X = np.random.randn(B, C, H, W)
    actual_output = instance_normalization(X, gamma, beta, num_groups)
    expected_output = [[[[ 0.90981361, -0.33429942],
                         [-0.16681702, -0.40869718]],
                        [[ 0.40560315, -0.22047364],
                         [-0.56160247,  0.37647295]]],
                       [[[-0.9411801,   0.60077322],
                         [ 0.48264645, -0.14223957]],
                        [[ 0.20283074, -0.38711656],
                         [-0.49567481,  0.67996064]]]]
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 2 failed")

    # Test different params
    gamma = np.ones(C).reshape(1, C, 1, 1) * 0.5
    beta = np.ones(C).reshape(1, C, 1, 1)
    actual_output = instance_normalization(X, gamma, beta, num_groups)
    expected_output = [[[[1.45490681, 0.83285029],
                         [0.91659149, 0.79565141]],
                        [[1.20280158, 0.88976318],
                         [0.71919877, 1.18823648]]],
                       [[[0.52940995, 1.30038661],
                         [1.24132322, 0.92888021]],
                        [[1.10141537, 0.80644172],
                         [0.75216259, 1.33998032]]]]
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, err_msg="Test case 3 failed")
    

if __name__ == "__main__":
    test_normalizations()
    print("All normalization tests passed.")
