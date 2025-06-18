## Understanding the Hard Sigmoid Activation Function

The Hard Sigmoid is a piecewise linear approximation of the sigmoid activation function. It's computationally more efficient than the standard sigmoid function while maintaining similar characteristics. This function is particularly useful in neural networks where computational efficiency is crucial.

### Mathematical Definition

The Hard Sigmoid function is mathematically defined as:

$$
HardSigmoid(x) = \begin{cases} 
0 & \text{if } x \leq -2.5 \\ 
0.2x + 0.5 & \text{if } -2.5 < x < 2.5 \\ 
1 & \text{if } x \geq 2.5 
\end{cases}
$$

Where $x$ is the input to the function.

### Characteristics

- **Output Range:** The output is always bounded in the range $[0, 1]$
- **Shape:** The function consists of three parts:
  - A constant value of 0 for inputs ≤ -2.5
  - A linear segment with slope 0.2 for inputs between -2.5 and 2.5
  - A constant value of 1 for inputs ≥ 2.5
- **Gradient:** The gradient is 0.2 in the linear region and 0 in the saturated regions

### Advantages in Neural Networks

This function is particularly useful in neural networks as it provides:
- Computational efficiency compared to standard sigmoid
- Bounded output range similar to sigmoid
- Simple gradient computation
