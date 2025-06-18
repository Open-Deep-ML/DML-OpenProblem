## Understanding the Cross Product

The **cross product** of two vectors $\vec{a}$ and $\vec{b}$ in 3D space is a vector that is perpendicular to both $\vec{a}$ and $\vec{b}$.

### Properties
- Defined only in 3 dimensions.
- The result $\vec{c} = \vec{a} \times \vec{b}$ is perpendicular to both $\vec{a}$ and $\vec{b}$.
- Follows the right-hand rule.

### Mathematical Formula

Given:
- $\vec{a} = [a_1, a_2, a_3]$
- $\vec{b} = [b_1, b_2, b_3]$

The cross product is:

$$
\vec{a} \times \vec{b} = [a_2 b_3 - a_3 b_2,\ a_3 b_1 - a_1 b_3,\ a_1 b_2 - a_2 b_1]
$$

### Use Cases
- Calculating normals in 3D graphics.
- Determining torque and angular momentum in physics.
- Verifying orthogonality in machine learning geometry.

### Example
For $\vec{a} = [1, 0, 0]$ and $\vec{b} = [0, 1, 0]$:

$$
\vec{a} \times \vec{b} = [0, 0, 1]
$$

The result points in the $z$-axis direction, confirming perpendicularity to both $\vec{a}$ and $\vec{b}$.
