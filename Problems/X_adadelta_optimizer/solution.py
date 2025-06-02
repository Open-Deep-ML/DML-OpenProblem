import numpy as np

def adadelta_optimizer(parameter, grad, u, v, rho=0.95, epsilon=1e-6):
    """
    Update parameters using the AdaDelta optimizer.
    AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
    monotonically decreasing learning rate.

    Args:
        parameter: Current parameter value
        grad: Current gradient
        u: Running average of squared gradients
        v: Running average of squared parameter updates
        rho: Decay rate for the moving average (default=0.95)
        epsilon: Small constant for numerical stability (default=1e-6)

    Returns:
        tuple: (updated_parameter, updated_u, updated_v)
    """
    assert 0 <= rho < 1, "Rho must be between 0 and 1"
    assert epsilon > 0, "Epsilon must be positive"
    assert all(u >= 0) if isinstance(u, np.ndarray) else u >= 0, "u must be non-negative"
    assert all(v >= 0) if isinstance(v, np.ndarray) else v >= 0, "v must be non-negative"

    # Update running average of squared gradients
    u = rho * u + (1 - rho) * grad**2

    # Compute RMS of gradient
    RMS_g = np.sqrt(u + epsilon)

    # Compute RMS of parameter updates
    RMS_dx = np.sqrt(v + epsilon)

    # Compute parameter update
    dx = -RMS_dx / RMS_g * grad

    # Update running average of squared parameter updates
    v = rho * v + (1 - rho) * dx**2

    # Update parameters
    parameter = parameter + dx

    return parameter, u, v

def test_adadelta_optimizer():
    """Test cases for the AdaDelta optimizer implementation."""
    rho = 0.95
    eps = 1e-6
    
    # Test case 1: Scalar inputs
    param = 1.0
    grad = 0.5
    u = 1.0
    v = 1.0

    new_param, new_u, new_v = adadelta_optimizer(param, grad, u, v, rho, eps)
    expected_u = rho * u + (1 - rho) * grad**2
    RMS_g = np.sqrt(expected_u + eps)
    RMS_dx = np.sqrt(v + eps)
    expected_dx = -RMS_dx / RMS_g * grad
    expected_v = rho * v + (1 - rho) * expected_dx**2
    expected_param = param + expected_dx

    assert np.isclose(new_param, expected_param), f"Unexpected parameter value: {new_param}"
    assert np.isclose(new_u, expected_u), f"Unexpected u value: {new_u}"
    assert np.isclose(new_v, expected_v), f"Unexpected v value: {new_v}"

    # Test case 2: Array inputs
    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])
    u = np.full_like(param, 1.0)
    v = np.full_like(param, 1.0)

    new_param, new_u, new_v = adadelta_optimizer(param, grad, u, v, rho, eps)
    expected_u = rho * u + (1 - rho) * grad**2
    RMS_g = np.sqrt(expected_u + eps)
    RMS_dx = np.sqrt(v + eps)
    expected_dx = -RMS_dx / RMS_g * grad
    expected_v = rho * v + (1 - rho) * expected_dx**2
    expected_param = param + expected_dx

    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_u, expected_u), f"Unexpected u values: {new_u}"
    assert np.allclose(new_v, expected_v), f"Unexpected v values: {new_v}"

    # Test case 3: Numeric stability
    param = np.array([1.0, 2.0])
    grad = np.array([0.0, 0.2])
    u = np.array([0.0, 1.0])
    v = np.array([0.0, 1.0])

    new_param, new_u, new_v = adadelta_optimizer(param, grad, u, v, rho, eps)
    expected_u = rho * u + (1 - rho) * grad**2
    RMS_g = np.sqrt(expected_u + eps)
    RMS_dx = np.sqrt(v + eps)
    expected_dx = -RMS_dx / RMS_g * grad
    expected_v = rho * v + (1 - rho) * expected_dx**2
    expected_param = param + expected_dx

    assert not np.isnan(new_u[0]), f"u should not be nan: {new_u}"
    assert not np.isnan(new_v[0]), f"v should not be nan: {new_v}"
    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_u, expected_u), f"Unexpected u values: {new_u}"
    assert np.allclose(new_v, expected_v), f"Unexpected v values: {new_v}"

    # Test case 4: Update is smaller for larger u
    param = np.array([1.0, 1.0])
    grad = np.array([1.0, 1.0])
    u = np.array([10000.0, 1.0])
    v = np.array([1.0, 1.0])

    new_param, new_u, new_v = adadelta_optimizer(param, grad, u, v, rho, eps)
    expected_u = rho * u + (1 - rho) * grad**2
    RMS_g = np.sqrt(expected_u + eps)
    RMS_dx = np.sqrt(v + eps)
    expected_dx = -RMS_dx / RMS_g * grad
    expected_v = rho * v + (1 - rho) * expected_dx**2
    expected_param = param + expected_dx

    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_u, expected_u), f"Unexpected u values: {new_u}"
    assert np.allclose(new_v, expected_v), f"Unexpected v values: {new_v}"
    # Verify that updates are much smaller for parameters with huge u
    assert np.abs(expected_dx[0]) < np.abs(expected_dx[1]), "Update should be smaller for larger u"

    # Test case 5: Zero rho
    param = 1.0
    grad = 0.5
    u = 1.0
    v = 1.0
    
    new_param, new_u, new_v = adadelta_optimizer(param, grad, u, v, 0, eps)
    expected_u = grad**2
    RMS_g = np.sqrt(expected_u + eps)
    RMS_dx = np.sqrt(v + eps)
    expected_dx = -RMS_dx / RMS_g * grad
    expected_v = expected_dx**2
    expected_param = param + expected_dx

    assert np.isclose(new_param, expected_param), f"Unexpected parameter value: {new_param}"
    assert np.isclose(new_u, expected_u), f"Unexpected u value: {new_u}"
    assert np.isclose(new_v, expected_v), f"Unexpected v value: {new_v}"

    print("All tests passed!")

if __name__ == "__main__":
    test_adadelta_optimizer()
