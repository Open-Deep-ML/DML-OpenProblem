# Implementing Momentum Optimizer

## Introduction
Momentum is a popular optimization technique that helps accelerate gradient descent in the relevant direction and dampens oscillations. It works by adding a fraction of the previous update vector to the current gradient.

## Learning Objectives
- Understand how momentum optimization works
- Learn to implement momentum-based gradient updates
- Understand the effect of momentum on optimization

## Theory
Momentum optimization uses a moving average of gradients to determine the direction of the update. The key equations are:

$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$ (Velocity update)

$\theta_t = \theta_{t-1} - v_t$ (Parameter update)

Where:
- $v_t$ is the velocity at time t
- $\gamma$ is the momentum coefficient (typically 0.9)
- $\eta$ is the learning rate
- $\nabla_\theta J(\theta)$ is the gradient of the loss function

Read more at:

1. Ruder, S. (2017). An overview of gradient descent optimization algorithms. [arXiv:1609.04747](https://arxiv.org/pdf/1609.04747)


## Problem Statement
Implement the momentum optimizer update step function. Your function should take the current parameter value, gradient, and velocity as inputs, and return the updated parameter value and new velocity.

### Input Format
The function should accept:
- parameter: Current parameter value
- grad: Current gradient
- velocity: Current velocity
- learning_rate: Learning rate (default=0.01)
- momentum: Momentum coefficient (default=0.9)

### Output Format
Return tuple: (updated_parameter, updated_velocity)

## Example
```python
# Example usage:
parameter = 1.0
grad = 0.1
velocity = 0.1

new_param, new_velocity = momentum_optimizer(parameter, grad, velocity)
```

## Tips
- Initialize velocity as zero
- Use numpy for numerical operations
- Test with both scalar and array inputs

---
