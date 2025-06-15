In this problem, you need to implement the Lasso Regression algorithm using Gradient Descent. Lasso Regression (L1 Regularization) adds a penalty equal to the absolute value of the coefficients to the loss function. Your task is to update the weights and bias iteratively using the gradient of the loss function and the L1 penalty.

The objective function of Lasso Regression is:
$$
J(w, b) = \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - \left( \sum_{j=1}^{p} X_{ij} w_j + b \right) \right)^2 + \alpha \sum_{j=1}^{p} | w_j |
$$

Where:
- $ y_i $ is the actual value for the $ i $-th sample
- $ \hat{y}_i = \sum_{j=1}^{p} X_{ij} w_j + b $ is the predicted value for the $ i $-th sample
- $ w_j $ is the weight associated with the $ j $-th feature
- $ \alpha $ is the regularization parameter
- $ b $ is the bias

Your task is to use the L1 penalty to shrink some of the feature coefficients to zero during gradient descent, thereby helping with feature selection.
