
# Understanding R-squared (R²) in Regression Analysis

R-squared, also known as the coefficient of determination, is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. It provides insight into how well the model fits the data.

### Mathematical Definition

The R-squared value is calculated using the following formula:  
$$
R^2 = 1 - \frac{\text{SSR}}{\text{SST}}
$$
Where:  

1) $ \text{SSR} $ (Sum of Squared Residuals): The sum of the squares of the differences between the actual values and the predicted values.  
2) $ \text{SST} $ (Total Sum of Squares): The sum of the squares of the differences between the actual values and the mean of the actual values.

### Equations for SSR and SST

To calculate SSR and SST, we use the following formulas:  

1) SSR:  
$$
\text{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$  

2) SST:  
$$
\text{SST} = \sum_{i=1}^{n} (y_i - \bar{y})^2
$$  

Where:  

1) $ y_i $: Actual value  
2) $ \hat{y}_i $: Predicted value  
3) $ \bar{y} $: Mean of the actual values  

### Significance of R-squared

R-squared is a key metric for evaluating how well a regression model performs. A higher R-squared value indicates a better fit for the model, meaning it can explain more variability in the data. However, it's important to note:  

- A high R-squared does not always imply that the model is good; it can sometimes be misleading if overfitting occurs.  
- It should be used in conjunction with other metrics for comprehensive model evaluation.

### Implementing R-squared Calculation

In this problem, you will implement a function to calculate R-squared given arrays of true and predicted values from a regression task. The results should be rounded to three decimal places.  

In the solution, the implemented $ r\_squared() $ function calculates R-squared by first determining SSR and SST, then applying them to compute $ R^2 $. It handles edge cases such as perfect predictions and situations where all true values are identical.

### Reference

You can refer to this resource for more information:  
[Coefficient of Determination](https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/coefficient-of-determination-r-squared.html)
