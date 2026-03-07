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
    # Your code here
    pass
