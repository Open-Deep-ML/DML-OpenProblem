## Phi Transformation

The Phi Transformation maps input features into a higher-dimensional space by generating polynomial features. This allows models like linear regression to fit nonlinear data by introducing new feature dimensions that represent polynomial combinations of the original input features.

### Why Use Phi Transformation?

- To increase the expressive power of simple models such as linear models.
- To enable better fitting of nonlinear relationships in the data.

### Equations

For an input value $x$, the Phi Transformation expands it as:

$$
\Phi(x) = [1, x, x^2, x^3, \dots, x^d]
$$

Where $d$ is the specified degree, and $\Phi(x)$ represents the transformed feature vector.

### Example 1: Polynomial Expansion for One Value

Given $x = 3$ and $d = 3$, the Phi Transformation is:

$$
\Phi(3) = [1, 3, 9, 27]
$$

### Example 2: Transformation for Multiple Values

For $\text{data} = [1, 2]$ and $d = 2$, the Phi Transformation is:

$$
\Phi([1, 2]) = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 4 \end{bmatrix}
$$
