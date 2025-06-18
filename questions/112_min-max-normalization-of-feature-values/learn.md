## Understanding Min-Max Normalization

Min-Max Normalization is a technique used to rescale numerical data to the range $[0, 1]$.

The formula used is:

$$
X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$

### Why Normalize?

- Ensures all features have equal importance regardless of their original scale.
- Commonly used in preprocessing for machine learning algorithms such as k-nearest neighbors, neural networks, and gradient descent-based models.

### Special Case

If all the elements in the input are identical, then $X_{\max} = X_{\min}$. In that case, return an array of zeros.

### Example

Given the input list `[1, 2, 3, 4, 5]`:

- Minimum: $1$
- Maximum: $5$
- The normalized values are:

$$
\begin{aligned}
&\frac{1 - 1}{4} = 0.0 \\
&\frac{2 - 1}{4} = 0.25 \\
&\frac{3 - 1}{4} = 0.5 \\
&\frac{4 - 1}{4} = 0.75 \\
&\frac{5 - 1}{4} = 1.0
\end{aligned}
$$

The result is `[0.0, 0.25, 0.5, 0.75, 1.0]`.

Remember to round the result to **4 decimal places**.
