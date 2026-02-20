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

    return np.round(parameter, 5), np.round(velocity, 5)
