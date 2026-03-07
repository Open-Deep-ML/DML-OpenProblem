import numpy as np

def fire_module_forward(input_tensor, 
                        squeeze_weights, squeeze_bias,
                        expand1x1_weights, expand1x1_bias,
                        expand3x3_weights, expand3x3_bias):
    """
    Implements the forward pass of a SqueezeNet Fire Module.
    
    Args:
        input_tensor: (H, W, C_in)
        squeeze_weights: (1, 1, C_in, s1x1)
        squeeze_bias: (s1x1,)
        expand1x1_weights: (1, 1, s1x1, e1x1)
        expand1x1_bias: (e1x1,)
        expand3x3_weights: (3, 3, s1x1, e3x3)
        expand3x3_bias: (e3x3,)
        
    Returns:
        output_tensor: (H, W, e1x1 + e3x3)
    
    Note:
    - All convolutions use stride=1.
    - 3x3 convolution uses padding='same' (pad=1).
    - Apply ReLU activation after each convolution (max(0, x)).
    """
    # Your code here
    pass
