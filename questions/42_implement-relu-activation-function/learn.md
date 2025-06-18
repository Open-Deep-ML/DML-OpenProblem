
## Understanding the ReLU Activation Function

The ReLU (Rectified Linear Unit) activation function is widely used in neural networks, particularly in hidden layers of deep learning models. It maps any real-valued number to the non-negative range $[0, \infty)$, which helps introduce non-linearity into the model while maintaining computational efficiency.

### Mathematical Definition
The ReLU function is mathematically defined as:
$$
f(z) = \max(0, z)
$$
where $z$ is the input to the function.

### Characteristics
- **Output Range**: The output is always in the range $[0, \infty)$. Values below 0 are mapped to 0, while positive values are retained.
- **Shape**: The function has an "L" shaped curve with a horizontal axis at $y = 0$ and a linear increase for positive $z$.
- **Gradient**: The gradient is 1 for positive values of $z$ and 0 for non-positive values. This means the function is linear for positive inputs and flat (zero gradient) for negative inputs.

This function is particularly useful in deep learning models as it introduces non-linearity while being computationally efficient, helping to capture complex patterns in the data.
