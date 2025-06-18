
## Understanding the Leaky ReLU Activation Function

The Leaky ReLU (Leaky Rectified Linear Unit) activation function is a variant of the ReLU function used in neural networks. It addresses the "dying ReLU" problem by allowing a small, non-zero gradient when the input is negative. This small slope for negative inputs helps keep the function active and prevents neurons from becoming inactive.

### Mathematical Definition
The Leaky ReLU function is mathematically defined as:
$$
f(z) = \begin{cases} 
z & \text{if } z > 0 \\ 
\alpha z & \text{if } z \leq 0 
\end{cases}
$$
where $z$ is the input to the function and $\alpha$ is a small positive constant, typically $\alpha = 0.01$.

In this definition, the function returns $z$ for positive values, and for negative values, it returns $\alpha z$, allowing a small gradient to pass through.

### Characteristics
- **Output Range**: The output is in the range $(-\infty, \infty)$. Positive values are retained, while negative values are scaled by the factor $\alpha$, allowing them to be slightly negative.
- **Shape**: The function has a similar "L" shaped curve as ReLU, but with a small negative slope on the left side for negative $z$, creating a small gradient for negative inputs.
- **Gradient**: The gradient is 1 for positive values of $z$ and $\alpha$ for non-positive values. This allows the function to remain active even for negative inputs, unlike ReLU, where the gradient is zero for negative inputs.

This function is particularly useful in deep learning models as it mitigates the issue of "dead neurons" in ReLU by ensuring that neurons can still propagate a gradient even when the input is negative, helping to improve learning dynamics in the network.
