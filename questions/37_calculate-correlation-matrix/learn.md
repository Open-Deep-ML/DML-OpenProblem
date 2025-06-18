
## Understanding Correlation Matrix

A correlation matrix is a table showing the correlation coefficients between variables. Each cell in the table shows the correlation between two variables, with values ranging from -1 to 1. These values indicate the strength and direction of the linear relationship between the variables.

### Mathematical Definition
The correlation coefficient between two variables \( X \) and \( Y \) is given by:
$$
\text{corr}(X, Y) = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}
$$

#### Where:
- \( \text{cov}(X, Y) \) is the covariance between \( X \) and \( Y \).
- \( \sigma_X \) and \( \sigma_Y \) are the standard deviations of \( X \) and \( Y \), respectively.

### Problem Overview
In this problem, you will write a function to calculate the correlation matrix for a given dataset. The function will take in a 2D numpy array \( X \) and an optional 2D numpy array \( Y \). If \( Y \) is not provided, the function will calculate the correlation matrix of \( X \) with itself.
