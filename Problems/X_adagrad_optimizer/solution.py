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

    return parameter, G

def test_adagrad_optimizer():
    """Test cases for the Adagrad optimizer implementation."""
    learning_rate = 0.01
    eps = 1e-8
    
    # Test case 1: Scalar inputs
    param = 1.0
    grad = 0.5
    G = 1.0

    new_param, new_G = adagrad_optimizer(param, grad, G, learning_rate, eps)
    expected_G = G + grad**2
    expected_update = learning_rate * grad / (np.sqrt(expected_G) + eps)
    expected_param = param - expected_update

    assert np.isclose(new_param, expected_param), f"Unexpected parameter value: {new_param}"
    assert np.isclose(new_G, expected_G), f"Unexpected G value: {new_G}"

    # Test case 2: Array inputs
    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])
    G = np.full_like(param, 1.0)

    new_param, new_G = adagrad_optimizer(param, grad, G, learning_rate, eps)
    expected_G = G + grad**2
    expected_updates = learning_rate * grad / (np.sqrt(expected_G) + eps)
    expected_param = param - expected_updates

    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_G, expected_G), f"Unexpected G values: {new_G}"

    # Test case 3: Numeric stability
    param = np.array([1.0, 2.0])
    grad = np.array([0.0, 0.2])
    G = np.array([0.0, 1.0])

    new_param, new_G = adagrad_optimizer(param, grad, G, learning_rate, eps)
    expected_G = G + grad**2
    expected_updates = learning_rate * grad / (np.sqrt(expected_G) + eps)
    expected_param = param - expected_updates

    assert not np.isnan(new_G[0]), f"G should not be nan: {new_G}"
    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_G, expected_G), f"Unexpected G values: {new_G}"

    # Test case 4: Update is smaller for larger G
    param = np.array([1.0, 1.0])
    grad = np.array([1.0, 1.0])
    G = np.array([10000.0, 1.0])

    new_param, new_G = adagrad_optimizer(param, grad, G, learning_rate, eps)
    expected_G = G + grad**2
    expected_updates = learning_rate * grad / (np.sqrt(expected_G) + eps)
    expected_param = param - expected_updates

    assert np.allclose(new_param, expected_param), f"Unexpected parameter values: {new_param}"
    assert np.allclose(new_G, expected_G), f"Unexpected G values: {new_G}"    
    # Verify that updates are much smaller for parameters with huge G
    assert np.abs(expected_updates[0]) < np.abs(expected_updates[1]), "Update should be smaller for larger G"

    print("All tests passed!")

if __name__ == "__main__":
    test_adagrad_optimizer()
