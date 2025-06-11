# Implementing Adagrad Optimizer

## Introduction
Adagrad (Adaptive Gradient Algorithm) is an optimization algorithm that adapts the learning rate to each parameter, performing larger updates for infrequent parameters and smaller updates for frequent ones. This makes it particularly well-suited for dealing with sparse data.

## Learning Objectives
- Understand how Adagrad optimizer works
- Learn to implement adaptive learning rates
- Gain practical experience with gradient-based optimization

## Theory
Adagrad adapts the learning rate for each parameter based on the historical gradients. The key equations are:

$G_t = G_{t-1} + g_t^2$ (Accumulated squared gradients)

$\theta_t = \theta_{t-1} - \dfrac{\alpha}{\sqrt{G_t} + \epsilon} \cdot g_t$ (Parameter update)

Where:
- $G_t$ is the sum of squared gradients up to time step t
- $\alpha$ is the initial learning rate
- $\epsilon$ is a small constant for numerical stability
- $g_t$ is the gradient at time step t

Read more at:

1. Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12, 2121–2159. [PDF](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
2. Ruder, S. (2017). An overview of gradient descent optimization algorithms. [arXiv:1609.04747](https://arxiv.org/pdf/1609.04747)


## Problem Statement
Implement the Adagrad optimizer update step function. Your function should take the current parameter value, gradient, and accumulated squared gradients as inputs, and return the updated parameter value and new accumulated squared gradients.

### Input Format
The function should accept:
- parameter: Current parameter value
- grad: Current gradient
- G: Accumulated squared gradients
- learning_rate: Learning rate (default=0.01)
- epsilon: Small constant for numerical stability (default=1e-8)

### Output Format
Return tuple: (updated_parameter, updated_G)

## Example
```python
# Example usage:
parameter = 1.0
grad = 0.1
G = 0.0

new_param, new_G = adagrad_optimizer(parameter, grad, G)
```

## Tips
- Initialize G as zeros
- Use numpy for numerical operations
- Test with both scalar and array inputs

---
