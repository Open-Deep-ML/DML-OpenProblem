### Mean Absolute Error (MAE)

The Mean Absolute Error (MAE) is a measure of the average magnitude of errors between predicted and actual values. Here's how to express it mathematically:

1. **Basic Formula**:
   - The MAE formula can be written as: $MAE = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$
   
   Where:
   - $n$ is the number of observations
   - $y_i$ is the true value
   - $\hat{y}_i$ is the predicted value
   - $|...|$ represents the absolute value

2. **Example Calculation**:
   For the values:
   ```
   y_true = [3, -0.5, 2, 7]
   y_pred = [2.5, 0.0, 2, 8]
   ```
   
   The calculation would be:
   $$
   \begin{align*}
   MAE &= \frac{1}{4}(|3-2.5| + |-0.5-0.0| + |2-2| + |7-8|) \\
   &= \frac{1}{4}(0.5 + 0.5 + 0 + 1) \\
   &= \frac{2}{4} \\
   &= 0.5
   \end{align*}
   $$

3. **Properties**:
   - MAE is always non-negative: $MAE \geq 0$
   - Perfect predictions result in $MAE = 0$
   - MAE is measured in the same units as the original data
   - MAE treats all errors with equal weight (unlike Mean Squared Error)

4. **Comparison with Other Metrics**:
   The formula for Mean Squared Error (MSE) is:
   $$MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$
   
   While MAE uses absolute values, MSE squares the differences, which:
   - Makes MSE more sensitive to outliers
   - Results in MSE values that are not in the original unit of measurement