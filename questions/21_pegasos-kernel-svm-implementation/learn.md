## Pegasos Algorithm for Kernel SVM (Deterministic Version)

### Introduction

The **Pegasos Algorithm** (Primal Estimated sub-GrAdient SOlver for SVM) is a fast, iterative algorithm designed to train Support Vector Machines (SVM). While the original Pegasos algorithm uses stochastic updates by selecting one random sample per iteration, this problem requires a **deterministic version**meaning **every data sample is evaluated and considered in each iteration**. This deterministic approach ensures reproducibility and clarity, particularly useful for educational purposes.

---

### Key Concepts

**Kernel Trick**:  
SVM typically separates data classes using a linear hyperplane. However, real-world data isn't always linearly separable. The **Kernel Trick** implicitly maps input data into a higher-dimensional feature space, making it easier to separate non-linear data.

Common kernel functions include:
- **Linear Kernel**: $K(x,y) = x \cdot y$
- **Radial Basis Function (RBF) Kernel**: $K(x,y) = e^{-\frac{\|x-y\|^2}{2\sigma^2}}$

**Regularization Parameter ($\lambda$)**:  
This parameter balances how closely the model fits training data against the complexity of the model, helping to prevent overfitting.

**Sub-gradient Descent**:  
Pegasos optimizes the SVM objective function using iterative parameter updates based on the sub-gradient of the hinge loss.

---

### Deterministic Pegasos Algorithm Steps

Given training samples $(x_i, y_i)$, labels $y_i \in \{-1, 1\}$, kernel function $K$, regularization parameter $\lambda$, and total iterations $T$:

1. **Initialize** alpha coefficients $\alpha_i = 0$ and bias $b = 0$.
2. For each iteration $t = 1, 2, \dots, T$:
    - Compute learning rate: $$\eta_t = \frac{1}{\lambda t}$$
    - For each training sample $(x_i, y_i)$:
        - Compute decision value:
        $$f(x_i) = \sum_{j}\alpha_j y_j K(x_j, x_i) + b$$
        - If the margin constraint $y_i f(x_i) < 1$ is violated, update parameters:
        $$
        \alpha_i \leftarrow \alpha_i + \eta_t(y_i - \lambda \alpha_i)
        $$
        $$
        b \leftarrow b + \eta_t y_i
        $$

---

### Example (Conceptual Explanation)

Consider a simple dataset:

- **Data**:  
$X = [[1,2],[2,3],[3,1],[4,1]]$, $Y = [1,1,-1,-1]$

- **Parameters**: Linear kernel, $\lambda = 0.01$, iterations = $1$

Initially, parameters ($\alpha, b$) start at zero. For each sample, you calculate the decision value. Whenever a sample violates the margin constraint ($y_i f(x_i) < 1$), you update the corresponding $\alpha_i$ and bias $b$ as described. After looping through all samples for the specified iterations, you obtain the trained parameters.

---

### Important Implementation Notes:
- Always iterate through **all samples** in every iteration (**no stochastic/random sampling**).
- Clearly distinguish kernel function choices in your implementation.
- After training, predictions for new data $x$ are made using:
$$
\hat{y}(x) = \text{sign}\left(\sum_{j}\alpha_j y_j K(x_j, x) + b\right)
$$

This deterministic Pegasos variant clearly demonstrates how kernelized SVM training operates and simplifies the understanding of kernel methods.
