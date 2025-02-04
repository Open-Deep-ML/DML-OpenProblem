## Understanding the Swish Activation Function

The Swish activation function is a modern self-gated activation function introduced by researchers at Google Brain. It has been shown to perform better than ReLU in many deep networks, particularly in deeper architectures.

### Mathematical Definition

The Swish function is defined as:

$$Swish(x) = x \times \sigma(x)$$

where $\sigma(x)$ is the sigmoid function defined as:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### Characteristics

- **Output Range**: Unlike ReLU which has a range of $[0, \infty)$, Swish has a range of $(-\infty, \infty)$
- **Smoothness**: Swish is smooth and non-monotonic, making it differentiable everywhere
- **Shape**: The function has a slight dip below 0 for negative values, then curves up smoothly for positive values
- **Properties**:
  - For large positive x: Swish(x) ≈ x (similar to linear function)
  - For large negative x: Swish(x) ≈ 0 (similar to ReLU)
  - Has a minimal value around x ≈ -1.28

### Advantages

- Smooth function with no hard zero threshold like ReLU
- Self-gated nature allows for more complex relationships
- Often provides better performance in deep neural networks
- Reduces the vanishing gradient problem compared to sigmoid
