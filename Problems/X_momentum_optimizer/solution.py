import numpy as np

def momentum_optimizer(parameter, grad, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the momentum optimizer.
    Uses momentum to accelerate learning in relevant directions and dampen oscillations.

    Args:
        parameter: Current parameter value
        grad: Current gradient
        velocity: Current velocity/momentum term
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)

    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
    assert learning_rate > 0, "Learning rate must be positive"
    assert 0 <= momentum < 1, "Momentum must be between 0 and 1"

    # Update velocity
    velocity = momentum * velocity + learning_rate * grad
    
    # Update parameters
    parameter = parameter - velocity

    return parameter, velocity

def test_momentum_optimizer():
    """Test cases for the momentum optimizer implementation."""
    learning_rate = 0.01
    momentum = 0.9

    # Test case 1: Scalar inputs
    param = 1.0
    grad = 0.1
    velocity = 0.5

    new_param, new_velocity = momentum_optimizer(param, grad, velocity, learning_rate, momentum)
    expected_velocity = momentum * velocity + learning_rate * grad
    expected_param = param - expected_velocity
    assert np.isclose(new_param, expected_param), f"Unexpected parameter value: {new_param}"
    assert np.isclose(new_velocity, expected_velocity), f"Unexpected velocity value: {new_velocity}"

    # Test case 2: Array inputs
    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])
    velocity = np.array([0.5, 1.0])

    new_param, new_velocity = momentum_optimizer(param, grad, velocity, learning_rate, momentum)
    expected_velocity = momentum * velocity + learning_rate * grad
    expected_param = param - expected_velocity

    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_velocity, expected_velocity), f"Unexpected velocity values: {new_velocity}"

    # Test case 3: Gradient update at zero momentum coefficient
    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])
    velocity = np.full_like(param, 0.5)

    new_param, new_velocity = momentum_optimizer(param, grad, velocity, learning_rate, 0)
    expected_velocity = learning_rate * grad
    expected_param = param - expected_velocity

    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"

    # Test case 4: Gradient update at zero gradient
    param = np.array([1.0, 2.0])
    grad = np.zeros_like(param)
    velocity = np.full_like(param, 0.5)
    
    new_param, new_velocity = momentum_optimizer(param, grad, velocity, learning_rate, momentum)
    expected_velocity = momentum * velocity
    expected_param = param - expected_velocity
    
    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_velocity, expected_velocity), f"Unexpected velocity values: {new_velocity}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_momentum_optimizer()
