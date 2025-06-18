
## Understanding The Conjugate Gradient Method

The Conjugate Gradient (CG) method is an iterative algorithm used to solve large systems of linear equations, particularly those that are symmetric and positive-definite.

### Concepts

The CG gradient method is often applied to the quadratic form of a linear system, $Ax = b$:

$$
f(x) = \frac{1}{2} x^T A x - b^T x
$$

The quadratic form is used due to its differential reducing to the following for a symmetric $A$. Therefore, $x$ satisfies $Ax = b$ at the optimum:

$$
f'(x) = Ax - b
$$

The conjugate gradient method uses search directions that are conjugate to all the previous search directions. This is satisfied when search directions are $A$-orthogonal, i.e.

$$
p_i^T A p_j = 0 \quad \text{for } i \neq j
$$

This results in a more efficient algorithm, as it ensures that the algorithm gathers all information in a search direction at once and then doesn't need to search in that direction again. This is opposed to steepest descent, where the algorithm steps a bit in one direction and then may search in that direction again later.

### Algorithm Steps

1) **Initialization**:
   - $ x_0 $: Initial guess for the variable vector.
   - $ r_0 = b - A x_0 $: Initial residual vector.
   - $ p_0 = r_0 $: Initial search direction.

2) **Iteration $k$**:
   - $ \alpha_k = \frac{r_k^T r_k}{p_k^T A p_k} $: Step size.
   - $ x_{k+1} = x_k + \alpha_k p_k $: Update solution.
   - $ r_{k+1} = r_k - \alpha_k A p_k $: Update residual.
   - Check convergence: $ \| r_{k+1} \| < \text{tolerance} $.
   - $ \beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k} $: New direction scaling. This ensures search directions are $A$-orthogonal.
   - $ p_{k+1} = r_{k+1} + \beta_k p_k $: Update search direction.

3) **Termination**:
   - Stop when $ \| r_{k+1} \| < \text{tolerance} $ or after a set number of iterations.

### Example Calculation

Let's solve the system of equations:

$$
4x_1 + x_2 = 6 \quad x_1 + 3x_2 = 6
$$

1) **Initialize**: $ x_0 = [0, 0]^T $, $ r_0 = b - A x_0 = [6, 6]^T $, and $ p_0 = r_0 = [6, 6]^T $.

2) **First iteration**:
   - Compute $ \alpha_0 $:

   $$
   \alpha_0 = \frac{r_0^T r_0}{p_0^T A p_0} = \frac{72}{324} = 0.2222
   $$

   - Update solution $ x_1 $:

   $$
   x_1 = x_0 + \alpha_0 p_0 = [0, 0]^T + 0.2222 \cdot [6, 6]^T = [1.3333, 1.3333]^T
   $$

   - Update residual $ r_1 $:

   $$
   r_1 = r_0 - \alpha_0 A p_0 = [6, 6]^T - 0.2222 \cdot \begin{bmatrix} 4 & 1 \\ 1 & 3 \end{bmatrix} \cdot [6, 6]^T = [6.67, 5.33]^T
   $$

   - Compute $ \beta_0 $:

   $$
   \beta_0 = \frac{r_1^T r_1}{r_0^T r_0} = \frac{6.67^2 + 5.33^2}{6^2 + 6^2} \approx 0.99
   $$

   - Update search direction $ p_1 $:

   $$
   p_1 = r_1 + \beta_0 p_0 = [6.67, 5.33]^T + 0.99 \cdot [6, 6]^T = [12.60, 11.26]^T
   $$

3) **Second iteration**:
   - Compute $ \alpha_1 $, $ x_2 $, $ r_2 $, and repeat until convergence.

### Applications

The conjugate gradient method is often used because it's more efficient than other iterative solvers, such as steepest descent, and direct solvers, such as Gaussian Elimination. Iterative linear solvers are commonly used in:

- Optimization
- Machine Learning
- Computational Fluid Dynamics
