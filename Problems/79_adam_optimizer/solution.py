import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using the Adam optimizer.
    Adjusts the learning rate based on the moving averages of the gradient and squared gradient.
    
    Args:
        parameter: Current parameter value (float or numpy array)
        grad: Current gradient (same shape as parameter)
        m: First moment estimate (same shape as parameter)
        v: Second moment estimate (same shape as parameter)
        t: Current timestep (integer)
        learning_rate: Learning rate (float, default=0.001)
        beta1: Decay rate for first moment (float, default=0.9)
        beta2: Decay rate for second moment (float, default=0.999)
        epsilon: Small constant for numerical stability (float, default=1e-8)
    
    Returns:
        tuple: (updated_parameter, updated_m, updated_v)
    """
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    parameter = parameter - update
    
    return parameter, m, v

def test_adam_optimizer():
    """Test cases for the Adam optimizer implementation."""
    # Test case 1: Scalar input
    param = 1.0
    grad = 0.1
    m = 0.0
    v = 0.0
    t = 1
    
    new_param, new_m, new_v = adam_optimizer(param, grad, m, v, t)
    assert isinstance(new_param, float), "Output parameter should be a float"
    assert isinstance(new_m, float), "Output m should be a float"
    assert isinstance(new_v, float), "Output v should be a float"
    
    # Test case 2: Array input
    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])
    m = np.zeros_like(param)
    v = np.zeros_like(param)
    t = 1
    
    new_param, new_m, new_v = adam_optimizer(param, grad, m, v, t)
    assert new_param.shape == param.shape, "Output parameter shape mismatch"
    assert new_m.shape == m.shape, "Output m shape mismatch"
    assert new_v.shape == v.shape, "Output v shape mismatch"
    
    # Test case 3: Check if updates are reasonable
    assert np.all(np.abs(new_param - param) < 1.0), "Update too large"
    assert np.all(new_m >= 0), "First moment should be non-negative"
    assert np.all(new_v >= 0), "Second moment should be non-negative"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_adam_optimizer()