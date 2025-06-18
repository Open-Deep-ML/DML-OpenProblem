
## Understanding Log Softmax Function

The log softmax function is a numerically stable way of calculating the logarithm of the softmax function. The softmax function converts a vector of arbitrary values (logits) into a vector of probabilities, where each value lies between 0 and 1, and the values sum to 1.

### Softmax Function
The softmax function is given by:
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$

### Log Softmax Function
Directly applying the logarithm to the softmax function can lead to numerical instability, especially when dealing with large numbers. To prevent this, we use the log-softmax function, which incorporates a shift by subtracting the maximum value from the input vector:
$$
\text{log softmax}(x_i) = x_i - \max(x) - \log\left(\sum_{j=1}^n e^{x_j - \max(x)}\right)
$$

This formulation helps to avoid overflow issues that can occur when exponentiating large numbers. The log-softmax function is particularly useful in machine learning for calculating probabilities in a stable manner, especially when used with cross-entropy loss functions.
