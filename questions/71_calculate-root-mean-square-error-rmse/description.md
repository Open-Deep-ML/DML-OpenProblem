
## Task: Compute Root Mean Square Error (RMSE)

In this task, you are required to implement a function `rmse(y_true, y_pred)` that calculates the Root Mean Square Error (RMSE) between the actual values and the predicted values. RMSE is a commonly used metric for evaluating the accuracy of regression models, providing insight into the standard deviation of residuals.

### Your Task:
Implement the function `rmse(y_true, y_pred)` to:
1. Calculate the RMSE between the arrays `y_true` and `y_pred`.
2. Return the RMSE value rounded to three decimal places.
3. Ensure the function handles edge cases such as:
   - Mismatched array shapes.
   - Empty arrays.
   - Invalid input types.

The RMSE is defined as:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}, i} - y_{\text{pred}, i})^2}
$$

Where:
- $ n $ is the number of observations.
- $ y_{\text{true}, i} $ and $ y_{\text{pred}, i} $ are the actual and predicted values for the $ i $-th observation.
