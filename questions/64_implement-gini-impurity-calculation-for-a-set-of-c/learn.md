
## Understanding Gini Impurity

Gini impurity is a statistical measurement of the impurity or disorder in a list of elements. It is commonly used in decision tree algorithms to decide the optimal split at tree nodes. It is calculated as follows, where $ p_i $ is the probability of each class, $ \frac{n_i}{n} $:

$$
\text{Gini Impurity} = 1 - \sum_{i=1}^{C} p_i^2
$$

A Gini impurity of 0 indicates a node where all elements belong to the same class, whereas a Gini impurity of 1-1/C indicates maximum impurity, where elements are evenly distributed among each class. This means that a lower impurity implies a less homogeneous distribution of elements, suggesting a good split, as decision trees aim to minimize it at each node.

### Advantages and Limitations

#### Advantages:
- Computationally efficient
- Works for binary and multi-class classification

#### Limitations:
- Biased toward larger classes
- May cause overfitting in deep decision trees

### Example Calculation

Suppose we have the set: $[0, 1, 1, 1, 0]$. The probability of each class is calculated as follows:

$$
p_{0} = \frac{2}{5} \quad p_{1} = \frac{3}{5}
$$

The Gini Impurity is then calculated as follows:

$$
\text{Gini Impurity} = 1 - (p_0^2 + p_1^2) = 1 - \left(\left(\frac{2}{5}\right)^2 + \left(\frac{3}{5}\right)^2\right) = 0.48
$$
