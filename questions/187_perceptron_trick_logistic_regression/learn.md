## Learning: Perceptron Trick in Logistic Regression

### Idea and formula
- **Goal**: Find a linear decision boundary that separates two classes using an iterative update rule.
- **Perceptron Learning Rule**: Update weights only when predictions are wrong.

The perceptron algorithm iteratively updates weights:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y_i \cdot \mathbf{x}_i \quad \text{if } y_i \cdot (\mathbf{w}^T \mathbf{x}_i) \leq 0
$$

Where:
- $\mathbf{w}$ is the weight vector (including bias)
- $\eta$ is the learning rate
- $y_i \in \{-1, +1\}$ is the true label
- $\mathbf{x}_i$ is the feature vector (with bias term added)

### Intuition
- When prediction is correct ($y_i \cdot (\mathbf{w}^T \mathbf{x}_i) > 0$): no update needed
- When prediction is wrong ($y_i \cdot (\mathbf{w}^T \mathbf{x}_i) \leq 0$): adjust weights to make the correct prediction more likely
- The update rule "pulls" the decision boundary toward misclassified points

### Algorithm steps
1. Initialize weights $\mathbf{w} = \mathbf{0}$ (or small random values)
2. Add bias term to each feature vector: $\mathbf{x}_i \leftarrow [\mathbf{x}_i, 1]$
3. For each epoch:
   - For each training example $(\mathbf{x}_i, y_i)$:
     - Compute prediction: $\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x}_i)$
     - If $y_i \cdot (\mathbf{w}^T \mathbf{x}_i) \leq 0$: update $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y_i \cdot \mathbf{x}_i$
4. Repeat until convergence or max epochs

### Convergence guarantee
- If data is linearly separable, perceptron algorithm will converge in finite steps
- Convergence time depends on the "margin" of separation

### Worked example
Given 2D data: $X = [[1,1], [2,2], [3,3]]$, $y = [1, 1, -1]$, $\eta = 0.1$:

- Start: $\mathbf{w} = [0, 0, 0]$ (including bias)
- Add bias: $X = [[1,1,1], [2,2,1], [3,3,1]]$

Epoch 1:
- $(1,1,1)$: $\mathbf{w}^T \mathbf{x} = 0$, $y \cdot (\mathbf{w}^T \mathbf{x}) = 0 \leq 0$ → update
  - $\mathbf{w} = [0,0,0] + 0.1 \cdot 1 \cdot [1,1,1] = [0.1, 0.1, 0.1]$
- $(2,2,1)$: $\mathbf{w}^T \mathbf{x} = 0.4$, $y \cdot (\mathbf{w}^T \mathbf{x}) = 0.4 > 0$ → no update
- $(3,3,1)$: $\mathbf{w}^T \mathbf{x} = 0.7$, $y \cdot (\mathbf{w}^T \mathbf{x}) = -0.7 \leq 0$ → update
  - $\mathbf{w} = [0.1,0.1,0.1] + 0.1 \cdot (-1) \cdot [3,3,1] = [-0.2, -0.2, 0.0]$

Continue until convergence...

### Edge cases and tips
- **Linearly separable data**: Algorithm will converge
- **Non-separable data**: May not converge; use max epochs limit
- **Learning rate**: Too large may cause oscillation; too small may converge slowly
- **Initialization**: Starting from zero is common; random initialization can help
- **Bias handling**: Always add bias term as additional feature
