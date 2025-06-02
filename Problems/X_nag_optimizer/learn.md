# Implementing Nesterov Accelerated Gradient (NAG) Optimizer

## Introduction
Nesterov Accelerated Gradient (NAG) is an improvement over classical momentum optimization. While momentum helps accelerate gradient descent in the relevant direction, NAG takes this a step further by looking ahead in the direction of the momentum before computing the gradient. This "look-ahead" property helps NAG make more informed updates and often leads to better convergence.

## Learning Objectives
- Understand how Nesterov Accelerated Gradient optimization works
- Learn to implement NAG-based gradient updates
- Understand the advantages of NAG over classical momentum
- Gain practical experience with advanced gradient-based optimization

## Theory
Nesterov Accelerated Gradient uses a "look-ahead" approach where it first makes a momentum-based step and then computes the gradient at that position. The key equations are:

$\theta_{lookahead, t-1} = \theta_{t-1} - \gamma v_{t-1}$ (Look-ahead position)

$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta_{lookahead, t-1})$ (Velocity update)

$\theta_t = \theta_{t-1} - v_t$ (Parameter update)

Where:
- $v_t$ is the velocity at time t
- $\gamma$ is the momentum coefficient (typically 0.9)
- $\eta$ is the learning rate
- $\nabla_\theta J(\theta)$ is the gradient of the loss function

The key difference from classical momentum is that the gradient is evaluated at $\theta_{lookahead, t-1}$ instead of $\theta_{t-1}$

## Problem Statement
Implement the Nesterov Accelerated Gradient optimizer update step function. Your function should take the current parameter value, gradient function, and velocity as inputs, and return the updated parameter value and new velocity.

### Input Format
The function should accept:
- parameter: Current parameter value
- gradient function: A function that accepts parameters and returns gradient computed at that point
- velocity: Current velocity
- learning_rate: Learning rate (default=0.01)
- momentum: Momentum coefficient (default=0.9)

### Output Format
Return tuple: (updated_parameter, updated_velocity)

## Example
```python
# Example usage:
def grad_func(parameter):
    # Returns gradient
    pass

parameter = 1.0
velocity = 0.0

new_param, new_velocity = nag_optimizer(parameter, grad_func, velocity)
```

## Tips
- Initialize velocity as zero
- Use numpy for numerical operations
- Test with both scalar and array inputs
- Remember that the gradient should be computed at the look-ahead position

---
