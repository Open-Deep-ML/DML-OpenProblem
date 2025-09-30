# **Gaussian Processes (GP): From-Scratch Regression Example**

## **1. What’s a Gaussian Process?**
A **Gaussian Process** defines a distribution over functions \( f(\cdot) \). For any finite set of inputs \(X=\{x_i\}_{i=1}^n\), the function values \(f(X)\) follow a multivariate normal:

\[
f(X) \sim \mathcal{N}\big(0,\; K(X,X)\big),
\]

where \(K\) is a **kernel** (covariance) function encoding similarity between inputs. With noisy targets \(y=f(X)+\varepsilon,\; \varepsilon\sim\mathcal{N}(0,\sigma_n^2 I)\), GP regression yields a closed-form posterior predictive mean and variance at new points \(X_*\).

---

## **2. The Implementation at a Glance**
The provided code builds a minimal yet complete GP regression stack:

- **Kernels implemented**
  - Radial Basis Function (RBF / Squared Exponential)
  - Matérn (\(\nu=0.5, 1.5, 2.5\), or general \(\nu\))
  - Periodic
  - Linear
  - Rational Quadratic
- **Core GP classes**
  - `_GaussianProcessBase`: kernel selection & covariance matrix computation
  - `GaussianProcessRegression`:
    - `fit`: builds \(K\), does **Cholesky decomposition**, solves \(\alpha\)
    - `predict`: returns posterior mean & variance
    - `log_marginal_likelihood`: computes GP evidence
    - `optimize_hyperparameters`: basic optimizer (for RBF hyperparams)

---

## **3. Kernel Cheat-Sheet**
Let \(x, x'\in\mathbb{R}^d\), \(r=\lVert x-x'\rVert\).

- **RBF (SE):**  
  \[
  k_{\text{RBF}}(x,x')=\sigma^2\exp\!\left(-\tfrac{1}{2}\tfrac{r^2}{\ell^2}\right)
  \]

- **Matérn (\(\nu=1.5\)):**  
  \[
  k(x,x')=\Big(1+\tfrac{\sqrt{3}\,r}{\ell}\Big)\exp\!\Big(-\tfrac{\sqrt{3}\,r}{\ell}\Big)
  \]

- **Periodic:**  
  \[
  k(x,x')=\sigma^2\exp\!\left(-\tfrac{2}{\ell^2}\sin^2\!\Big(\tfrac{\pi r}{p}\Big)\right)
  \]

- **Linear:**  
  \[
  k(x,x')=\sigma_b^2+\sigma_v^2\,x^\top x'
  \]

- **Rational Quadratic:**  
  \[
  k(x,x')=\sigma^2\Big(1+\tfrac{r^2}{2\alpha \ell^2}\Big)^{-\alpha}
  \]

---

## **4. GP Regression Mechanics**
### Training
1. Build covariance:  
   \(K = K(X,X) + \sigma_n^2 I\)
2. Cholesky factorization:  
   \(K=LL^\top\)
3. Solve \(\alpha\):  
   \(L L^\top \alpha = y\)

### Prediction
At new inputs \(X_*\):
- \(K_* = K(X, X_*)\), \(K_{**} = K(X_*, X_*)\)
- **Mean:**  
  \(\mu_* = K_*^\top \alpha\)
- **Covariance:**  
  \(\Sigma_* = K_{**} - V^\top V,\;\; V = L^{-1}K_*\)

### Model Selection
- **Log Marginal Likelihood (LML):**  
  \[
  \log p(y\mid X)= -\tfrac{1}{2}y^\top \alpha - \sum\nolimits_i \log L_{ii} - \tfrac{n}{2}\log(2\pi)
  \]

---

## **5. Worked Example (Linear Kernel)**

```python
import numpy as np
gp = GaussianProcessRegression(kernel='linear',
                               kernel_params={'sigma_b': 0.0, 'sigma_v': 1.0},
                               noise=1e-8)

X_train = np.array([[1], [2], [4]])
y_train = np.array([3, 5, 9])   # y = 2x + 1
gp.fit(X_train, y_train)

X_test = np.array([[3.0]])
mu = gp.predict(X_test)
print(f"{mu[0]:.4f}")   # -> 7.0000
```


## **6. When to Use GP Regression**

- **Small-to-medium datasets** where uncertainty estimates are valuable  
- Cases requiring **predictive intervals** (not just point predictions)  
- **Nonparametric modeling** with kernel priors  
- Automatic hyperparameter tuning via **marginal likelihood**  

---

## **7. Practical Tips**

- Always add **jitter** (`1e-6`) to the diagonal for numerical stability  
- **Standardize inputs/outputs** before training  
- Be aware: Exact GP has complexity **\(\mathcal{O}(n^3)\)** in time and **\(\mathcal{O}(n^2)\)** in memory  
- Choose kernels to match problem structure:  
  - **RBF:** smooth functions  
  - **Matérn:** rougher functions  
  - **Periodic:** seasonal/cyclical data  
  - **Linear:** global linear trends  

