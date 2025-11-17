### Problem

Implement simple linear regression using Ordinary Least Squares (OLS). Given 1D inputs `X` and targets `y`, compute the slope `m`, intercept `b`, and use them to predict on a provided test input.

You should implement the closed-form OLS solution:

$$
m = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2},\quad
b = \bar{y} - m\,\bar{x}.
$$

Then, given `X_test`, output predictions `y_pred = m * X_test + b`.

Return values: `m`, `b`, and `y_pred`.

