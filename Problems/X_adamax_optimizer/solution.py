import numpy as np

def adamax_optimizer(parameter, grad, m, u, t, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using the Adamax optimizer.
    Adamax is a variant of Adam based on the infinity norm.
    It uses the maximum of past squared gradients instead of the exponential moving average.

    Args:
        parameter: Current parameter value
        grad: Current gradient
        m: First moment estimate
        u: Infinity norm estimate
        t: Current timestep
        learning_rate: Learning rate (default=0.002)
        beta1: First moment decay rate (default=0.9)
        beta2: Infinity norm decay rate (default=0.999)
        epsilon: Small constant for numerical stability (default=1e-8)

    Returns:
        tuple: (updated_parameter, updated_m, updated_u)
    """
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Update infinity norm estimate
    u = np.maximum(beta2 * u, np.abs(grad))

    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1**t)

    # Update parameters
    update = learning_rate * m_hat / (u + epsilon)
    parameter = parameter - update

    return parameter, m, u

def test_adamax_optimizer():
    """Test cases for the Adamax optimizer implementation."""
    learning_rate = 0.002
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Test case 1: Scalar inputs
    param = 1.0
    grad = 0.1
    m = 1.0
    u = 1.0
    t = 1

    new_param, new_m, new_u = adamax_optimizer(param, grad, m, u, t, learning_rate, beta1, beta2, epsilon)
    expected_m = beta1 * m + (1 - beta1) * grad
    bias_corrected_m = expected_m / (1 - beta1**t)
    expected_u = np.maximum(beta2 * u, np.abs(grad))
    expected_param = param - learning_rate * bias_corrected_m / (expected_u + epsilon)

    assert np.isclose(new_param, expected_param), f"Unexpected parameter value: {new_param}"
    assert np.isclose(new_m, expected_m), f"Unexpected m value: {new_m}"
    assert np.isclose(new_u, expected_u), f"Unexpected u value: {new_u}"

    # Test case 2: Array inputs
    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])
    m = np.full_like(param, 1.0)
    u = np.full_like(param, 1.0)
    t = 1

    new_param, new_m, new_u = adamax_optimizer(param, grad, m, u, t, learning_rate, beta1, beta2, epsilon)
    expected_m = beta1 * m + (1 - beta1) * grad
    bias_corrected_m = expected_m / (1 - beta1**t)
    expected_u = np.maximum(beta2 * u, np.abs(grad))
    expected_param = param - learning_rate * bias_corrected_m / (expected_u + epsilon)

    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_m, expected_m), f"Unexpected m values: {new_m}"
    assert np.allclose(new_u, expected_u), f"Unexpected u values: {new_u}"

    # Test case 3: Numerical stability
    param = np.array([1.0, 2.0])
    grad = np.array([0.0, 0.0])
    m = np.full_like(param, 1.0)
    u = np.zeros_like(param)
    t = 1

    new_param, new_m, new_u = adamax_optimizer(param, grad, m, u, t, learning_rate, beta1, beta2, epsilon)
    expected_m = beta1 * m + (1 - beta1) * grad
    bias_corrected_m = expected_m / (1 - beta1**t)
    expected_u = np.maximum(beta2 * u, np.abs(grad))
    expected_param = param - learning_rate * bias_corrected_m / (expected_u + epsilon)

    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_m, expected_m), f"Unexpected m values: {new_m}"
    assert np.allclose(new_u, expected_u), f"Unexpected u values: {new_u}"

    # Test case 4: Zero betas
    param = 1.0
    grad = 0.1
    m = 1.0
    u = 1.0
    t = 1

    new_param, new_m, new_u = adamax_optimizer(param, grad, m, u, t, learning_rate, 0, 0, epsilon)
    expected_m = grad
    expected_u = np.abs(grad)
    expected_param = param - learning_rate * expected_m / (expected_u + epsilon)
    assert np.isclose(new_param, expected_param), f"Unexpected parameter value: {new_param}"
    assert np.isclose(new_m, expected_m), f"Unexpected m value: {new_m}"
    assert np.isclose(new_u, expected_u), f"Unexpected u value: {new_u}"


    print("All tests passed!")

if __name__ == "__main__":
    test_adamax_optimizer()
