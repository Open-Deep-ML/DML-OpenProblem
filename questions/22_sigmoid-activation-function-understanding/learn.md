
## Understanding the Sigmoid Activation Function

The sigmoid activation function is crucial in neural networks, especially for binary classification tasks. It maps any real-valued number into the interval \( (0, 1) \), making it useful for modeling probability as an output.

### Mathematical Definition
The sigmoid function is mathematically defined as:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
where \( z \) is the input to the function.

### Characteristics
- **Output Range**: The output is always between 0 and 1.
- **Shape**: The function has an "S" shaped curve.
- **Gradient**: The gradient is highest near \( z = 0 \) and decreases as \( z \) moves away from 0 in either direction.

The sigmoid function is particularly useful for turning logits (raw prediction values) into probabilities in binary classification models.
