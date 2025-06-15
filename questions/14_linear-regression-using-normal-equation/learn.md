
## Linear Regression Using the Normal Equation

Linear regression aims to model the relationship between a scalar dependent variable \( y \) and one or more explanatory variables (or independent variables) \( X \). The normal equation provides an analytical solution to find the coefficients \( \theta \) that minimize the cost function for linear regression.

Given a matrix \( X \) (with each row representing a training example and each column a feature) and a vector \( y \) (representing the target values), the normal equation is:
$$
\theta = (X^TX)^{-1}X^Ty
$$

### Explanation of Terms
1. \( X^T \) is the transpose of \( X \).
2. \( (X^TX)^{-1} \) is the inverse of the matrix \( X^TX \).
3. \( y \) is the vector of target values.

### Key Points
- **Feature Scaling**: This method does not require feature scaling.
- **Learning Rate**: There is no need to choose a learning rate.
- **Computational Cost**: Computing the inverse of \( X^TX \) can be computationally expensive if the number of features is very large.

### Practical Implementation
A practical implementation involves augmenting \( X \) with a column of ones to account for the intercept term and then applying the normal equation directly to compute \( \theta \).
