# Implementing Adamax Optimizer

## Introduction
Adamax is a variant of Adam optimizer that uses the infinity norm (max) instead of the L2 norm for the second moment estimate. This makes it more robust in some cases and can lead to better convergence in certain scenarios, particularly when dealing with sparse gradients.

## Learning Objectives
- Understand how Adamax optimization works
- Learn to implement Adamax-based gradient updates
- Understand the effect of infinity norm on optimization

## Theory
Adamax maintains a moving average of gradients (first moment) and uses the infinity norm for the second moment estimate. The key equations are:

First moment estimate (same as Adam):

$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$

The second moment estimate in Adam uses the $l_2$ norm:

$v_t = \beta_2 v_{t-1} + (1-\beta_2)|g_t|^2$

This can be generalized to the $l_p$ norm, but norms for large p values are numerically unstable. However, Adamax uses the $l_\infin$ norm (infinity norm), which converges to:

$u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$

Unlike Adam, Adamax doesn't require bias correction for $u_t$ because the max operation makes it less susceptible to bias towards zero.

Bias correction:
$\hat{m}_t = \dfrac{m_t}{1-\beta_1^t}$

Parameter update:
$\theta_t = \theta_{t-1} - \dfrac{\eta}{u_t} \hat{m}_t$

Where:
- $m_t$ is the first moment estimate at time t
- $u_t$ is the infinity norm estimate at time t
- $\beta_1$ is the first moment coefficient (typically 0.9)
- $\beta_2$ is the second moment coefficient (typically 0.999)
- $\eta$ is the learning rate
- $g_t$ is the gradient at time t

Note: Unlike Adam, Adamax doesn't require bias correction for $u_t$ because the max operation makes it less susceptible to bias towards zero.

Read more at:

1. Kingma, D. and Ba, J. (2015). Adam: A Method for Stochastic Optimization. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
2. Ruder, S. (2017). An overview of gradient descent optimization algorithms. [arXiv:1609.04747](https://arxiv.org/pdf/1609.04747)


## Problem Statement
Implement the Adamax optimizer update step function. Your function should take the current parameter value, gradient, and moment estimates as inputs, and return the updated parameter value and new moment estimates.

### Input Format
The function should accept:
- parameter: Current parameter value
- grad: Current gradient
- m: First moment estimate
- u: Infinity norm estimate
- t: Current timestep
- learning_rate: Learning rate (default=0.002)
- beta1: First moment decay rate (default=0.9)
- beta2: Second moment decay rate (default=0.999)
- epsilon: Small constant for numerical stability (default=1e-8)

### Output Format
Return tuple: (updated_parameter, updated_m, updated_u)

## Example
```python
# Example usage:
parameter = 1.0
grad = 0.1
m = 0.0
u = 0.0
t = 1

new_param, new_m, new_u = adamax_optimizer(parameter, grad, m, u, t)
```

## Tips
- Initialize m and u as zeros
- Keep track of timestep t for bias correction
- Use numpy for numerical operations
- Test with both scalar and array inputs
- Remember to apply bias correction to the first moment estimate

---
