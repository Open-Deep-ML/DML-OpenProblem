import numpy as np

def rmsprop_optimizer(parameter, grad, v, learning_rate=0.001, beta=0.9, epsilon=1e-8):
    """
    Update parameters using the RMSprop optimizer.
    Adjusts the learning rate based on the moving average of squared gradients.
    
    :param parameter: Current parameter value
    :param grad: Current gradient
    :param v: Moving average of squared gradients
    :param learning_rate: Learning rate (default=0.001)
    :param beta: Decay rate for squared gradient moving average (default=0.9)
    :param epsilon: Small constant for numerical stability (default=1e-8)
    :return: tuple: (updated_parameter, updated_v)
    """
    # Update moving average of squared gradients
    v = beta * v + (1 - beta) * (grad ** 2)
    
    # Update parameter using adjusted learning rate
    parameter = parameter - learning_rate * grad / (np.sqrt(v) + epsilon)
    
    return np.round(parameter, 5), np.round(v, 5)

