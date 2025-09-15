## Learning: PMF normalization constant

### Idea and formula
- **PMF requirement**: A probability mass function must satisfy ∑ p(xᵢ) = 1 and p(xᵢ) ≥ 0.
- **Normalization by a constant**: If probabilities are given up to a constant, you determine that constant by enforcing the sum-to-1 constraint.
  - If the form is p(xᵢ) = K · wᵢ with known nonnegative weights wᵢ, then
    - ∑ p(xᵢ) = K · ∑ wᵢ = 1 ⇒ **K = 1 / ∑ wᵢ**.
  - If the given expressions involve K in a more general way (e.g., both K and K² terms), still enforce ∑ p(xᵢ) = 1 and solve the resulting equation for K. Choose the solution that makes all probabilities nonnegative.

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

These add up to 9K + 10K² as required.

Enforce the PMF constraint:

- 9K + 10K² = 1 ⇒ 10K² + 9K − 1 = 0

Quadratic formula reminder:

- For aK² + bK + c = 0, the solutions are K = [−b ± √(b² − 4ac)] / (2a).

Solve the quadratic:

- K = [−9 ± √(9² + 4·10·1)] / (2·10) = [−9 ± √121] / 20 = [−9 ± 11] / 20
- Feasible (K ≥ 0) root: K = (−9 + 11) / 20 = 2/20 = 0.1

Therefore, the normalization constant is **K = 0.1**.
