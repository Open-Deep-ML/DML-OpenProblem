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
    assert learning_rate > 0, "Learning rate must be positive"
    assert 0 <= beta1 < 1, "Beta1 must be between 0 and 1"
    assert 0 <= beta2 < 1, "Beta2 must be between 0 and 1"
    assert epsilon > 0, "Epsilon must be positive"
    assert all(m >= 0) if isinstance(m, np.ndarray) else m >= 0, "m must be non-negative"
    assert all(u >= 0) if isinstance(u, np.ndarray) else u >= 0, "u must be non-negative"

    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Update infinity norm estimate
    u = np.maximum(beta2 * u, np.abs(grad))

    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1**t)

    # Update parameters
    update = learning_rate * m_hat / (u + epsilon)
    parameter = parameter - update

    return np.round(parameter, 5), np.round(m, 5), np.round(u, 5)