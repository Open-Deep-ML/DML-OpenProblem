## Understanding the Gram-Schmidt Process

The Gram-Schmidt process transforms a set of vectors into an orthonormal basis vectors that are orthogonal (perpendicular) and have unit length for the subspace they span.

### Mathematical Definition

Given vectors $v_1, v_2, \ldots$, the process constructs an orthonormal set $u_1, u_2, \ldots$ as follows:
1. $u_1 = \frac{v_1}{\|v_1\|}$ (normalize the first vector).
2. For subsequent vectors $v_k$:
   - Subtract projections: $$w_k = v_k - \sum_{i=1}^{k-1} \text{proj}_{u_i}(v_k),$$ where $\text{proj}_{u_i}(v_k) = (v_k \cdot u_i) u_i$.
   - Normalize: $$u_k = \frac{w_k}{\|w_k\|},$$ if $\|w_k\| > \text{tol}$.

### Why Orthonormal Bases?

- Orthogonal vectors simplify computations (e.g., their dot product is zero).
- Unit length ensures equal scaling, useful in $PCA$, $QR$ decomposition, and neural network optimization.

### Special Case

If a vector's norm is less than or equal to $\text{tol}$ (default $1e-10$), it's considered linearly dependent and excluded from the basis.

### Example

For vectors `[[1, 0], [1, 1]]` with $\text{tol} = 1e-10$:
1. $v_1 = [1, 0]$, $\|v_1\| = 1$, so $u_1 = [1, 0]$.
2. $v_2 = [1, 1]$, projection on $u_1$: $(v_2 \cdot u_1) u_1 = 1 \cdot [1, 0] = [1, 0]$.
   - $w_2 = [1, 1] - [1, 0] = [0, 1]$.
   - $\|w_2\| = 1 > 1e-10$, so $u_2 = [0, 1]$.

Result: `[[1, 0], [0, 1]]`, rounded to 4 decimal places.
