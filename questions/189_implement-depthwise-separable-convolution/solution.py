import numpy as np

def depthwise_separable_conv2d(
    input: np.ndarray,
    depthwise_filters: np.ndarray,
    pointwise_filters: np.ndarray
) -> np.ndarray:
    """
    Implements depthwise separable convolution.
    
    Args:
        input: Input tensor of shape (H, W, C_in)
        depthwise_filters: Depthwise filters of shape (K, K, C_in)
        pointwise_filters: Pointwise filters of shape (1, 1, C_in, C_out)
    
    Returns:
        Output tensor of shape (H_out, W_out, C_out)
        where H_out = H - K + 1 and W_out = W - K + 1 (assuming stride=1, no padding)
    """
    H, W, C_in = input.shape
    K, _, _ = depthwise_filters.shape
    _, _, _, C_out = pointwise_filters.shape
    
    # Calculate output dimensions
    H_out = H - K + 1
    W_out = W - K + 1
    
    # Step 1: Depthwise convolution
    # Apply one filter per input channel independently
    depthwise_output = np.zeros((H_out, W_out, C_in))
    
    for h in range(H_out):
        for w in range(W_out):
            for c in range(C_in):
                # Extract patch for current channel
                patch = input[h:h+K, w:w+K, c]
                # Apply corresponding depthwise filter
                depthwise_output[h, w, c] = np.sum(patch * depthwise_filters[:, :, c])
    
    # Step 2: Pointwise convolution (1x1 convolution)
    # Mix channels to produce final output
    output = np.zeros((H_out, W_out, C_out))
    
    for h in range(H_out):
        for w in range(W_out):
            for k in range(C_out):
                # Combine all input channels for output channel k
                output[h, w, k] = np.sum(depthwise_output[h, w, :] * pointwise_filters[0, 0, :, k])
    
    return output
