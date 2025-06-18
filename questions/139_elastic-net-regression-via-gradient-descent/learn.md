# Elastic Net Regression Using Gradient Descent

Elastic Net Regression combines both L1 (Lasso) and L2 (Ridge) regularization techniques to overcome the limitations of using either regularization method alone. It's particularly useful when dealing with datasets that have many correlated features.

## What is Elastic Net?

Elastic Net addresses two main issues:
- **Lasso's limitation**: When features are highly correlated, Lasso tends to select only one feature from a group of correlated features arbitrarily
- **Ridge's limitation**: Ridge regression doesn't perform feature selection (coefficients approach zero but never become exactly zero)

The goal of Elastic Net is to minimize the objective function:

$$J(w, b) = \underbrace{\frac{1}{2n} \sum_{i=1}^n\left( y_i - \left(\sum_{j=1}^pX_{ij}w_j+b\right)\right)^2}_{\text{MSE Loss}} + \underbrace{\alpha_1 \sum_{j=1}^p |w_j|}_{\text{L1 Regularization}} + \underbrace{\alpha_2 \sum_{j=1}^p w_j^2}_{\text{L2 Regularization}}$$

Where:
* The first term is the **Mean Squared Error (MSE) Loss**: $\frac{1}{2n} \sum_{i=1}^n\left( y_i - \left(\sum_{j=1}^pX_{ij}w_j+b\right)\right)^2$
* The second term is the **L1 Regularization** (Lasso penalty): $\alpha_1 \sum_{j=1}^p |w_j|$
* The third term is the **L2 Regularization** (Ridge penalty): $\alpha_2 \sum_{j=1}^p w_j^2$
* $\alpha_1$ controls the strength of L1 regularization
* $\alpha_2$ controls the strength of L2 regularization

## Step-by-Step Implementation Guide

### 1. Initialize weights $w_j$ and bias $b$ to 0

### 2. Make Predictions
At each iteration, calculate predictions using:
$$\hat{y}_i = \sum_{j=1}^pX_{ij}w_j + b$$

Where:
- $\hat{y}_i$ is the predicted value for the $i$-th sample
- $X_{ij}$ is the value of the $i$-th sample's $j$-th feature
- $w_j$ is the weight associated with the $j$-th feature

### 3. Calculate Residuals
Find the difference between actual and predicted values: $error_i = \hat{y}_i - y_i$

### 4. Update Weights and Bias Using Gradients

**Gradient with respect to weights:**
$$\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^nX_{ij}(\hat{y}_i - y_i) + \alpha_1 \cdot \text{sign}(w_j) + 2\alpha_2 \cdot w_j$$

**Gradient with respect to bias:**
$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^n(\hat{y}_i - y_i)$$

**Update rules:**
$$w_j = w_j - \eta \cdot \frac{\partial J}{\partial w_j}$$
$$b = b - \eta \cdot \frac{\partial J}{\partial b}$$

Where $\eta$ is the learning rate.

### 5. Check for Convergence
Repeat steps 2-4 until convergence. Convergence is determined by evaluating the L1 norm of the weight gradients:

$$||\nabla w||_1 = \sum_{j=1}^p \left|\frac{\partial J}{\partial w_j}\right|$$

If $||\nabla w||_1 < \text{tolerance}$, stop the algorithm.

### 6. Return the Final Weights and Bias

## Key Parameters

- **alpha1**: L1 regularization strength (promotes sparsity)
- **alpha2**: L2 regularization strength (handles correlated features)
- **learning_rate**: Step size for gradient descent
- **max_iter**: Maximum number of iterations
- **tol**: Convergence tolerance
Path
## Key Differences from Lasso and Ridge

1. **Lasso (L1 only)**: Tends to select one feature from correlated groups, can be unstable with small sample sizes
2. **Ridge (L2 only)**: Keeps all features but shrinks coefficients, doesn't perform feature selection
3. **Elastic Net (L1 + L2)**: Combines benefits of both - performs feature selection while handling correlated features better than Lasso alone

The balance between L1 and L2 regularization is controlled by the `alpha1` and `alpha2` parameters, allowing you to tune the model for your specific dataset characteristics.
