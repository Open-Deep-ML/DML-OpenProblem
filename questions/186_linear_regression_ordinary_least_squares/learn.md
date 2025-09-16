## Learning: Ordinary Least Squares for Simple Linear Regression

### Idea and formula
- **Goal**: Fit a line $y = m x + b$ that minimizes the sum of squared errors.
- **Closed-form OLS solution** for 1D features:

$$
m = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2},\quad
b = \bar{y} - m\,\bar{x}
$$

### Intuition
- The numerator is the sample covariance between $x$ and $y$; the denominator is the sample variance of $x$.
- So $m = \operatorname{Cov}(x,y) / \operatorname{Var}(x)$ measures how much $y$ changes per unit change in $x$.
- The intercept $b$ anchors the best-fit line so it passes through the mean point $(\bar{x},\bar{y})$.

### Algorithm steps
1. Compute $\bar{x}$ and $\bar{y}$.
2. Accumulate numerator $\sum_i (x_i-\bar{x})(y_i-\bar{y})$ and denominator $\sum_i (x_i-\bar{x})^2$.
3. Compute $m = \text{numerator}/\text{denominator}$ (guard against zero denominator).
4. Compute $b = \bar{y} - m\,\bar{x}$.
5. Predict: $\hat{y} = m\,x + b$ for any new $x$.

### Edge cases and tips
- If all $x_i$ are identical, $\operatorname{Var}(x)=0$ and the slope is undefined. In practice, return $m=0$ and $b=\bar{y}$ or raise an error.
- Centering data helps numerical stability but is not required for the closed form.
- Outliers can strongly influence OLS; consider robust alternatives if needed.

### Worked example
Given $X = [1,2,3]$ and $y = [2,2.5,3.5]$:

- $\bar{x} = 2$, $\bar{y} = 8/3$.
- $\sum (x_i-\bar{x})(y_i-\bar{y}) = (1-2)(2-8/3) + (2-2)(2.5-8/3) + (3-2)(3.5-8/3) = 1.5$
- $\sum (x_i-\bar{x})^2 = (1-2)^2 + (2-2)^2 + (3-2)^2 = 2$
- $m = 1.5/2 = 0.75$
- $b = \bar{y} - m\,\bar{x} = 8/3 - 0.75\cdot 2 = 1.166666\ldots$

Prediction for $X_{test} = [4]$: $y_{pred} = 0.75\cdot 4 + 1.1666\ldots = 4.1666\ldots$

