
## Root Mean Square Error (RMSE)

RMSE is used to measure the accuracy of predictions in regression models. It represents the difference between the predictions and the actual values. In other words, it is the standard deviation of the residuals or prediction errors.

### **Theory**
The RMSE is defined as:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}_i} - y_{\text{pred}_i})^2}
$$

where:
- $ n $: The number of observations.
- $ y_{\text{true}_i} $: The actual values.
- $ y_{\text{pred}_i} $: The predicted values.

### **Steps for Calculation**
1. For each pair of actual and predicted values, calculate the difference $ y_{\text{true}_i} - y_{\text{pred}_i} $.
2. Square each of these differences and find their mean.
3. Take the square root of the mean value.

### **When to Use RMSE vs. MAE**
- **RMSE**: Used when large deviations/errors are more problematic and should be penalized more heavily.
- **MAE**: Used when errors should be treated equally, regardless of their size.
