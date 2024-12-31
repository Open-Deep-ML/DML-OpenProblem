import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using the Adam optimizer.
    Adjusts the learning rate based on the moving averages of the gradient and squared gradient.

    Args:
        parameter: Current parameter value
        grad: Current gradient
        m: First moment estimate
        v: Second moment estimate
        t: Current timestep
        learning_rate: Learning rate (default=0.001)
        beta1: First moment decay rate (default=0.9)
        beta2: Second moment decay rate (default=0.999)
        epsilon: Small constant for numerical stability (default=1e-8)

    Returns:
        tuple: (updated_parameter, updated_m, updated_v)
    """
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * (grad**2)

    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1**t)

    # Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - beta2**t)

    # Update parameters
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    parameter = parameter - update

    return parameter, m, v

def test_adam_optimizer():
    """Test cases for the Adam optimizer implementation."""
    # Test case 1: Scalar inputs
    param = 1.0
    grad = 0.1
    m = 0.0
    v = 0.0
    t = 1

    new_param, new_m, new_v = adam_optimizer(param, grad, m, v, t)
    assert np.isclose(new_param, 0.999), f"Unexpected parameter value: {new_param}"
    assert np.isclose(new_m, 0.01), f"Unexpected m value: {new_m}"
    assert np.isclose(new_v, 0.0001), f"Unexpected v value: {new_v}"

    # Test case 2: Array inputs
    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])
    m = np.zeros_like(param)
    v = np.zeros_like(param)

    new_param, new_m, new_v = adam_optimizer(param, grad, m, v, t)
    expected_param = np.array([0.999, 1.999])
    expected_m = np.array([0.01, 0.02])
    expected_v = np.array([0.0001, 0.0004])

    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_m, expected_m), f"Unexpected m values: {new_m}"
    assert np.allclose(new_v, expected_v), f"Unexpected v values: {new_v}"

    print("All tests passed!")

if __name__ == "__main__":
    test_adam_optimizer()
