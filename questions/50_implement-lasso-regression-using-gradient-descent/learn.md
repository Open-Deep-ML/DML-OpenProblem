## Understanding Lasso Regression and L1 Regularization

Lasso Regression is a type of linear regression that applies L1 regularization to the model. It adds a penalty equal to the sum of the absolute values of the coefficients, encouraging some of them to be exactly zero. This makes Lasso Regression particularly useful for feature selection, as it can shrink the coefficients of less important features to zero, effectively removing them from the model.

### Steps to Implement Lasso Regression using Gradient Descent

1. **Initialize Weights and Bias**:  
   Start with the weights and bias set to zero.

2. **Make Predictions**:  
   Use the formula:
   $$
   \hat{y}_i = \sum_{j=1}^{p} X_{ij} w_j + b
   $$
   where $ \hat{y}_i $ is the predicted value for the $ i $-th sample.

3. **Compute Residuals**:  
   Find the difference between the predicted values $ \hat{y}_i $ and the actual values $ y_i $. These residuals are the errors in the model.

4. **Update the Weights and Bias**:  
   Update the weights and bias using the gradient of the loss function with respect to the weights and bias:

   1. For weights $ w_j $:
      $$
      \frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} X_{ij}(\hat{y}_i - y_i) + \alpha \cdot \text{sign}(w_j)
      $$

   2. For bias $ b $ (without the regularization term):
      $$
      \frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
      $$

   3. Update the weights and bias:
      $$
      w_j = w_j - \eta \cdot \frac{\partial J}{\partial w_j}
      $$
      $$
      b = b - \eta \cdot \frac{\partial J}{\partial b}
      $$

5. **Check for Convergence**:  
   The algorithm stops when the L1 norm of the gradient with respect to the weights becomes smaller than a predefined threshold $ \text{tol} $:
   $$
   ||\nabla w ||_1 = \sum_{j=1}^{p} \left| \frac{\partial J}{\partial w_j} \right|
   $$

6. **Return the Weights and Bias**:  
   Once the algorithm converges, return the optimized weights and bias.
