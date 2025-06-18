## Binary Classification with Logistic Regression

Logistic Regression is a fundamental algorithm for binary classification. Given input features and learned model parameters (weights and bias), your task is to implement the prediction function that computes class probabilities.

### Mathematical Background

The logistic regression model makes predictions using the sigmoid function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where z is the linear combination of features and weights plus bias:

$$z = \mathbf{w}^T\mathbf{x} + b = \sum_{i=1}^{n} w_ix_i + b$$

### Implementation Requirements

Your task is to implement a function that:

- Takes a batch of samples $\mathbf{X}$ (shape: N x D), weights $\mathbf{w}$ (shape: D), and bias b
- Computes $z = \mathbf{X}\mathbf{w} + b$ for all samples
- Applies the sigmoid function to get probabilities
- Returns binary predictions i.e., 0 or 1 using a threshold of 0.5

### Important Considerations

- Handle numerical stability in sigmoid computation
- Ensure efficient vectorized operations using numpy
- Return binary predictions (0 or 1)

### Hint

To prevent overflow in the exponential calculation of the sigmoid function, use `np.clip` to limit z values:

```python
z = np.clip(z, -500, 500)
```
This ensures numerical stability when dealing with large input values.
