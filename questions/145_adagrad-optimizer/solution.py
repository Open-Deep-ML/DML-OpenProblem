import numpy as np

def adagrad_optimizer(parameter, grad, G, learning_rate=0.01, epsilon=1e-8):
    """
    Update parameters using the Adagrad optimizer.
    Adapts the learning rate for each parameter based on the historical gradients.

    Args:
        parameter: Current parameter value
        grad: Current gradient
        G: Accumulated squared gradients
        learning_rate: Learning rate (default=0.01)
        epsilon: Small constant for numerical stability (default=1e-8)

    Returns:
        tuple: (updated_parameter, updated_G)
    """
    assert learning_rate > 0, "Learning rate must be positive"
    assert epsilon > 0, "Epsilon must be positive"
    assert all(G >= 0) if isinstance(G, np.ndarray) else G >= 0, "G must be non-negative"

    # Update accumulated squared gradients
    G = G + grad**2

    # Update parameters using adaptive learning rate
    update = learning_rate * grad / (np.sqrt(G) + epsilon)
    parameter = parameter - update

    return np.round(parameter, 5), np.round(G, 5)
