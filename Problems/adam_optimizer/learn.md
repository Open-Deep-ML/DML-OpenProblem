# Implementing Adam Optimizer

## Introduction
Adam (Adaptive Moment Estimation) is a popular optimization algorithm used in training deep learning models. It combines the benefits of two other optimization algorithms: RMSprop and momentum optimization.

## Learning Objectives
- Understand how Adam optimizer works
- Learn to implement adaptive learning rates
- Understand bias correction in optimization algorithms
- Gain practical experience with gradient-based optimization

## Theory
Adam maintains moving averages of both gradients (first moment) and squared gradients (second moment) to adapt the learning rate for each parameter. The key equations are:

$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ (First moment)
$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ (Second moment)

Bias correction:
$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$
$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$

Parameter update:
$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

## Problem Statement
Implement the Adam optimizer update step function. Your function should take the current parameter value, gradient, and moving averages as inputs, and return the updated parameter value and new moving averages.

### Input Format
The function should accept:
- parameter: Current parameter value
- grad: Current gradient
- m: First moment estimate
- v: Second moment estimate
- t: Current timestep
- learning_rate: Learning rate (default=0.001)
- beta1: First moment decay rate (default=0.9)
- beta2: Second moment decay rate (default=0.999)
- epsilon: Small constant for numerical stability (default=1e-8)

### Output Format
Return tuple: (updated_parameter, updated_m, updated_v)

## Example
```python
# Example usage:
parameter = 1.0
grad = 0.1
m = 0.0
v = 0.0
t = 1

new_param, new_m, new_v = adam_optimizer(parameter, grad, m, v, t)
```

## Tips
- Initialize m and v as zeros
- Keep track of timestep t for bias correction
- Use numpy for numerical operations
- Test with both scalar and array inputs

---
