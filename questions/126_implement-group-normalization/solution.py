def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    '''
    Perform Group Normalization.
    
    Args:
    X: numpy array of shape (B, C, H, W), input data
    gamma: numpy array of shape (C,), scale parameter
    beta: numpy array of shape (C,), shift parameter
    num_groups: number of groups for normalization
    epsilon: small constant to avoid division by zero
    
    Returns:
    norm_X: numpy array of shape (B, C, H, W), normalized output
    '''
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
