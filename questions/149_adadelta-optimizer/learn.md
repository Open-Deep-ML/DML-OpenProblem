# Implementing Adadelta Optimizer

## Introduction
Adadelta is an extension of Adagrad that addresses two key issues: the aggressive, monotonically decreasing learning rate and the need for manual learning rate tuning. While Adagrad accumulates all past squared gradients, Adadelta restricts the influence of past gradients to a window of size w. Instead of explicitly storing w past gradients, it efficiently approximates this window using an exponential moving average with decay rate ρ, making it more robust to parameter updates. Additionally, it automatically handles the units of the updates, eliminating the need for a manually set learning rate.

## Learning Objectives
- Understand how Adadelta optimizer works
- Learn to implement adaptive learning rates with moving averages

## Theory
Adadelta uses two main ideas:
1. Exponential moving average of squared gradients to approximate a window of size w
2. Automatic unit correction through the ratio of parameter updates

The key equations are:

$v_t = \rho v_{t-1} + (1-\rho)g_t^2$ (Exponential moving average of squared gradients)

The above approximates a window size of $w \approx \dfrac{1}{1-\rho}$ 

$\Delta\theta_t = -\dfrac{\sqrt{v_{t-1} + \epsilon}}{\sqrt{u_t + \epsilon}} \cdot g_t$ (Parameter update with unit correction)

$u_t = \rho u_{t-1} + (1-\rho)\Delta\theta_t^2$ (Exponential moving average of squared parameter updates)

Where:
- $v_t$ is the exponential moving average of squared **parameter updates** (decay rate ρ)
- $u_t$ is the exponential moving average of squared **gradients** (decay rate ρ)
- $\rho$ is the decay rate (typically 0.9) that controls the effective window size w ≈ 1/(1-ρ)
- $\epsilon$ is a small constant for numerical stability
- $g_t$ is the gradient at time step t

The ratio $\dfrac{\sqrt{v_{t-1} + \epsilon}}{\sqrt{u_t + \epsilon}}$ serves as an adaptive learning rate that automatically handles the units of the updates, making the algorithm more robust to different parameter scales. Unlike Adagrad, Adadelta does not require a manually set learning rate, making it especially useful when tuning hyperparameters is difficult. This automatic learning rate adaptation is achieved through the ratio of the root mean squared (RMS) of parameter updates to the RMS of gradients.

Read more at:

1. Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. [arXiv:1212.5701](https://arxiv.org/abs/1212.5701)
2. Ruder, S. (2017). An overview of gradient descent optimization algorithms. [arXiv:1609.04747](https://arxiv.org/pdf/1609.04747)

## Problem Statement
Implement the Adadelta optimizer update step function. Your function should take the current parameter value, gradient, and accumulated statistics as inputs, and return the updated parameter value and new accumulated statistics.

### Input Format
The function should accept:
- parameter: Current parameter value
- grad: Current gradient
- u: Exponentially decaying average of squared gradients
- v: Exponentially decaying average of squared parameter updates
- rho: Decay rate (default=0.9)
- epsilon: Small constant for numerical stability (default=1e-8)

### Output Format
Return tuple: (updated_parameter, updated_v, updated_u)

## Example
```python
# Example usage:
parameter = 1.0
grad = 0.1
v = 1.0
u = 1.0

new_param, new_v, new_u = adadelta_optimizer(parameter, grad, v, u)
```

## Tips
- Initialize v and u as zeros
- Use numpy for numerical operations
- Test with both scalar and array inputs
- The learning rate is automatically determined by the algorithm

---