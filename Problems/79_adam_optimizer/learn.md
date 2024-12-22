# File: learn.md

# Implementing Adam Optimizer

## Introduction
Adam (Adaptive Moment Estimation) is one of the most popular optimization algorithms in deep learning. It combines the advantages of two other extensions of stochastic gradient descent: RMSprop and momentum optimization. Adam adapts the learning rates of each parameter by using estimates of first and second moments of the gradients.

## Learning Objectives
- Understand the components of the Adam optimization algorithm
- Learn how momentum and RMSprop are combined in Adam
- Implement bias correction in optimization
- Gain practical experience with numerical optimization techniques

## Theory

### Adam Algorithm
Adam maintains two moving averages:
1. First moment (mean) of gradients (m)
2. Second moment (uncentered variance) of gradients (v)

The algorithm updates these moving averages and uses them to adapt the learning rates for each parameter:

$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$

where:
- $g_t$ is the gradient at time step t
- $\beta_1$ and $\beta_2$ are decay rates for the moving averages

To counteract the bias toward zero in the early steps, Adam uses bias-corrected versions of the moments:

$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$
$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$

The parameter update rule is then:

$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

where:
- $\alpha$ is the learning rate
- $\epsilon$ is a small constant for numerical stability

## Problem Statement
Implement the Adam optimizer update step function. The function should take the current parameter value, gradient, and moving averages as inputs, and return the updated parameter value and new moving averages.

### Input Format
- parameter: Current parameter value (float or numpy array)
- grad: Current gradient (same shape as parameter)
- m: First moment estimate (same shape as parameter)
- v: Second moment estimate (same shape as parameter)
- t: Current timestep (integer)
- learning_rate: Learning rate (float, default=0.001)
- beta1: Decay rate for first moment (float, default=0.9)
- beta2: Decay rate for second moment (float, default=0.999)
- epsilon: Small constant for numerical stability (float, default=1e-8)

### Output Format
Return a tuple containing:
- Updated parameter value
- Updated first moment estimate (m)
- Updated second moment estimate (v)

## Examples

### Example 1
```python
# Input
parameter = 1.0
grad = 0.1
m = 0.0
v = 0.0
t = 1

# Output
parameter_new, m_new, v_new = adam_optimizer(parameter, grad, m, v, t)
# parameter_new ≈ 0.999
# m_new ≈ 0.01
# v_new ≈ 0.01
```

## Tips
- Initialize m and v as zeros with the same shape as your parameters
- Keep track of the timestep t for bias correction
- Use numpy for array operations
- Be careful with the order of operations to avoid numerical instability
- Test your implementation with both scalar and array inputs

---