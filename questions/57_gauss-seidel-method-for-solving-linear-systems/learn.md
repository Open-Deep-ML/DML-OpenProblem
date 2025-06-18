
## Understanding the Gauss-Seidel Method

The Gauss-Seidel method is a technique for solving linear systems of equations $Ax = b$. Unlike fixed-point Jacobi, Gauss-Seidel uses previously computed results as soon as they are available. This increases convergence, resulting in fewer iterations, but it is not as easily parallelizable as fixed-point Jacobi.

### Mathematical Formulation

1. **Initialization**: Start with an initial guess for $x$.

2. **Iteration**: For each equation $i$, update $x[i]$ using:

$$
x_{i}^{(k+1)} = \frac{1}{a_{ii}} \left( b[i] - \sum_{j < i} a_{ij} x_{j}^{(k+1)} - \sum_{j > i} a_{ij} x_{j}^{(k)} \right)
$$

where $a_{ii}$ represents the diagonal elements of $A$, and $a_{ij}$ represents the off-diagonal elements.

3. **Convergence**: Repeat the iteration until the changes in $x$ are below a set tolerance or until a maximum number of iterations is reached.

### Matrix Form

The Gauss-Seidel method can also be expressed in matrix form using the diagonal matrix $D$, lower triangle $L$, and upper triangle $U$:

$$
x^{(k+1)} = D^{-1} \left( b - Lx^{(k+1)} - Ux^{(k)} \right)
$$

### Example Calculation

Let’s solve the system of equations:

$$
3x_1 + x_2 = 5 \quad x_1 + 2x_2 = 5
$$

1. Initialize $ x_1^{(0)} = 0 $ and $ x_2^{(0)} = 0 $.

2. **First iteration**:

For $ x_1^{(1)} $:

$$
x_1^{(1)} = \frac{1}{3} \left( 5 - 1 \cdot x_2^{(0)} \right) = \frac{5}{3} \approx 1.6667
$$

For $ x_2^{(1)} $:

$$
x_2^{(1)} = \frac{1}{2} \left( 5 - 1 \cdot x_1^{(1)} \right) = \frac{1}{2} \left( 5 - 1.6667 \right) \approx 1.6667
$$

After the first iteration, the values are $ x_1^{(1)} = 1.6667 $ and $ x_2^{(1)} = 1.6667 $.

Continue iterating until the results converge to a desired tolerance.

### Applications

The Gauss-Seidel method and other iterative solvers are commonly used in data science, computational fluid dynamics, and 3D graphics.
