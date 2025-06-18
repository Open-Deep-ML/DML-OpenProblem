## Understanding Vector Element-wise Sum

In linear algebra, the **element-wise sum** (also known as vector addition) involves adding corresponding entries of two vectors.

### Vector Notation
Given two vectors $a$ and $b$ of the same dimension $n$:

$$
a = \begin{pmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix}, \quad b = \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{pmatrix}
$$

The element-wise sum is defined as:

$$
a + b = \begin{pmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{pmatrix}
$$

### Key Requirement
Vectors $a$ and $b$ must be of the same length $n$ for the operation to be valid. If their lengths differ, element-wise addition is not defined.

### Example
Let:

$$
a = [1, 2, 3], \quad b = [4, 5, 6]
$$

Then:

$$
a + b = [1+4, 2+5, 3+6] = [5, 7, 9]
$$

This simple operation is foundational in many applications such as vector arithmetic, neural network computations, and linear transformations.
