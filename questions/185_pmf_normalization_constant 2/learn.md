## Learning: PMF normalization constant

### Idea and formula
- **PMF requirement**: A probability mass function must satisfy $\sum_i p(x_i) = 1$ and $p(x_i) \ge 0$.
- **Normalization by a constant**: If probabilities are given up to a constant, determine that constant by enforcing the sum-to-1 constraint.
  - If the form is $p(x_i) = K\,w_i$ with known nonnegative weights $w_i$, then
    - $\sum_i p(x_i) = K \sum_i w_i = 1 \Rightarrow$ $\displaystyle K = \frac{1}{\sum_i w_i}$.
  - If the given expressions involve $K$ in a more general way (e.g., both $K$ and $K^2$ terms), still enforce $\sum_i p(x_i) = 1$ and solve the resulting equation for $K$. Choose the solution that makes all probabilities nonnegative.

### Worked example (this question)
Suppose the PMF entries are expressed in terms of K such that, when summed, the K-terms group as follows:

- Linear in K: K + 2K + 2K + 3K + K = 9K
- Quadratic in K: K² + 2K² + 7K² = 10K²

One concrete way to realize this is via the following table of outcomes and probabilities:

| X  | p(X)         |
|----|--------------|
| x₁ | K + K²       |
| x₂ | 2K + 2K²     |
| x₃ | 2K           |
| x₄ | 3K + 7K²     |
| x₅ | K            |

These add up to $9K + 10K^2$ as required.

Enforce the PMF constraint:

$$
9K + 10K^2 = 1 \;\Rightarrow\; 10K^2 + 9K - 1 = 0
$$

Quadratic formula reminder:

$$
\text{For } aK^2 + bK + c = 0,\quad K = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}.
$$

Solve the quadratic:

$$
K = \frac{-9 \pm \sqrt{9^2 - 4\cdot 10 \cdot (-1)}}{2\cdot 10}
= \frac{-9 \pm \sqrt{121}}{20}
= \frac{-9 \pm 11}{20}.
$$

Feasible ($K \ge 0$) root: $\displaystyle K = \frac{2}{20} = 0.1$.

Therefore, the normalization constant is **$K = 0.1$**.
