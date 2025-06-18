## Calculating the Dot Product of Two Vectors

The dot product, also known as the scalar product, is a mathematical operation that takes two equal-length vectors and returns a single number. It is widely used in physics, geometry, and linear algebra.

### 1. Formula for the Dot Product

The dot product of two vectors $\mathbf{a}$ and $\mathbf{b}$, each of length $n$, is calculated as follows:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i
$$

This means multiplying corresponding elements of the two vectors and summing up the results.

### 2. Geometric Interpretation

In geometric terms, the dot product can also be expressed as:

$$
\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}| |\mathbf{b}| \cos \theta
$$

Where:

- $|\mathbf{a}|$ and $|\mathbf{b}|$ are the magnitudes of the vectors.
- $\theta$ is the angle between the two vectors.

### 3. Properties of the Dot Product

1. **Commutative**:  
   $$
   \mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a}
   $$

2. **Distributive**:  
   $$
   \mathbf{a} \cdot (\mathbf{b} + \mathbf{c}) = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{c}
   $$

3. **Orthogonal Vectors**:  
   If:  
   $$
   \mathbf{a} \cdot \mathbf{b} = 0
   $$  
   Then $\mathbf{a}$ and $\mathbf{b}$ are perpendicular.

### 4. Example Calculation

Given two vectors:

- $\mathbf{a} = [1, 2, 3]$
- $\mathbf{b} = [4, 5, 6]$

The dot product is calculated as:

$$
\mathbf{a} \cdot \mathbf{b} = (1 \cdot 4) + (2 \cdot 5) + (3 \cdot 6) = 4 + 10 + 18 = 32
$$

### Conclusion

The dot product is a fundamental operation in vector algebra, useful in determining angles between vectors, projections, and in many applications across physics and engineering.
