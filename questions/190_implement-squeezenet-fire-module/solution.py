import numpy as np

def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)

def conv2d(x, w, b, padding=0):
    """
    Apply 2D convolution.
    x: (H, W, C_in)
    w: (kh, kw, C_in, C_out)
    b: (C_out,)
    padding: int (symmetrical padding on H and W)
    """
    H, W, C_in = x.shape
    kh, kw, _, C_out = w.shape
    
    # Apply padding if needed
    if padding > 0:
        x_padded = np.pad(x, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        x_padded = x
    
    # Output dimensions (assuming stride=1)
    H_out = H
    W_out = W
    
    output = np.zeros((H_out, W_out, C_out))
    
    # Optimization for 1x1 convolution
    if kh == 1 and kw == 1:
        # Flatten spatial dimensions: (H*W, C_in)
        x_flat = x.reshape(-1, C_in)
        # Weights: (C_in, C_out)
        w_flat = w.reshape(C_in, C_out)
        # Matrix multiplication + bias
        out_flat = np.dot(x_flat, w_flat) + b
        return out_flat.reshape(H, W, C_out)
    
    # General convolution (loops)
    for h in range(H_out):
        for w_idx in range(W_out):
            # Extract patch
            patch = x_padded[h:h+kh, w_idx:w_idx+kw, :]
            # Convolution for each output channel
            for c in range(C_out):
                kernel = w[:, :, :, c]
                output[h, w_idx, c] = np.sum(patch * kernel) + b[c]
                
    return output

def fire_module_forward(input_tensor, 
                        squeeze_weights, squeeze_bias,
                        expand1x1_weights, expand1x1_bias,
                        expand3x3_weights, expand3x3_bias):
    """
    Implements the forward pass of a SqueezeNet Fire Module.
    """
    # 1. Squeeze Layer (1x1 conv)
    squeeze_out = conv2d(input_tensor, squeeze_weights, squeeze_bias)
    squeeze_act = relu(squeeze_out)
    
    # 2. Expand Layer
    # Branch 1: 1x1 conv
    expand1x1_out = conv2d(squeeze_act, expand1x1_weights, expand1x1_bias)
    expand1x1_act = relu(expand1x1_out)
    
    # Branch 2: 3x3 conv (with padding='same' -> pad=1)
    expand3x3_out = conv2d(squeeze_act, expand3x3_weights, expand3x3_bias, padding=1)
    expand3x3_act = relu(expand3x3_out)
    
    # 3. Concatenate along channel axis
    output = np.concatenate([expand1x1_act, expand3x3_act], axis=-1)
    
    return output
