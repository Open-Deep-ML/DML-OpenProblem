import numpy as np

def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the Nesterov Accelerated Gradient optimizer.
    Uses a "look-ahead" approach to improve convergence by applying momentum before computing the gradient.

    Args:
        parameter: Current parameter value
        grad_fn: Function that computes the gradient at a given position
        velocity: Current velocity (momentum term)
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)

    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
    assert 0 <= momentum < 1, "Momentum must be between 0 and 1"
    assert learning_rate > 0, "Learning rate must be positive"

    # Compute look-ahead position
    look_ahead = parameter - momentum * velocity
    
    # Compute gradient at look-ahead position
    grad = grad_fn(look_ahead)
    
    # Update velocity using momentum and gradient
    velocity = momentum * velocity + learning_rate * grad
    
    # Update parameters using the new velocity
    parameter = parameter - velocity

    return parameter, velocity

def test_nag_optimizer():
    """Test cases for the Nesterov Accelerated Gradient optimizer implementation."""
    learning_rate = 0.01
    momentum = 0.9

    def gradient_function(x):
        """ Gradient of the function f(x) = (x - np.arange(len(x)))^2 / 2"""
        if isinstance(x, np.ndarray):
            n = len(x)
            return x - np.arange(n)
        else:
            return x - 0

    # Test case 1: Scalar inputs
    param = 1.0
    velocity = 0.5

    new_param, new_velocity = nag_optimizer(param, gradient_function, velocity, learning_rate, momentum)
    look_ahead = param - momentum * velocity
    grad = gradient_function(look_ahead)
    expected_velocity = momentum * velocity + learning_rate * grad
    expected_param = param - expected_velocity
    assert np.isclose(new_param, expected_param), f"Unexpected parameter value: {new_param}"
    assert np.isclose(new_velocity, expected_velocity), f"Unexpected velocity value: {new_velocity}"

    # Test case 2: Array inputs
    param = np.array([1.0, 2.0])
    velocity = np.array([0.5, 1.0])

    new_param, new_velocity = nag_optimizer(param, gradient_function, velocity, learning_rate, momentum)
    look_ahead = param - momentum * velocity
    grad = gradient_function(look_ahead)
    expected_velocity = momentum * velocity + learning_rate * grad
    expected_param = param - expected_velocity

    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_velocity, expected_velocity), f"Unexpected velocity values: {new_velocity}"

    # Test case 3: Gradient update at zero momentum coefficient
    param = np.array([1.0, 2.0])
    velocity = np.full_like(param, 0.5)

    new_param, new_velocity = nag_optimizer(param, gradient_function, velocity, learning_rate, 0)
    grad = gradient_function(param)
    expected_velocity = learning_rate * grad
    expected_param = param - expected_velocity

    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"

    # Test case 4: Gradient update when look-ahead gradient is zero
    velocity = 1
    param = velocity * momentum

    new_param, new_velocity = nag_optimizer(param, gradient_function, velocity, learning_rate, momentum)
    expected_velocity = momentum * velocity
    expected_param = np.zeros_like(param)

    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_velocity, expected_velocity), f"Unexpected velocity values: {new_velocity}"

    print("All tests passed!")

if __name__ == "__main__":
    test_nag_optimizer()