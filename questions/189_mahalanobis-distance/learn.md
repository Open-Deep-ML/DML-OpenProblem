## Mahalanobis Distance

The Mahalanobis distance is a fundamental metric in multivariate statistics and machine learning used to measure the distance between a point and a probability distribution. Unlike Euclidean distance, which treats all dimensions equally, the Mahalanobis distance accounts for the correlations and scales of different features via the covariance matrix.

### Formula

The Mahalanobis distance from a point $x$ to a distribution with mean $\mu$ and covariance matrix $\Sigma$ is defined as:

$$D_M = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

### Components

- **$(x - \mu)$**: The deviation vector from the point to the mean
- **$\Sigma^{-1}$**: The inverse of the covariance matrix, which accounts for variance and correlation
- **$(x - \mu)^T \Sigma^{-1} (x - \mu)$**: A quadratic form that measures standardized distance

### Properties

1. **Scale-Invariant**: Takes into account the variance along each dimension
2. **Correlation-Aware**: Accounts for correlations between features through the covariance matrix
3. **Special Case**: When the covariance matrix is the identity matrix, Mahalanobis distance reduces to Euclidean distance
4. **Outlier Detection**: Points with large Mahalanobis distance are statistical outliers relative to the distribution

### Applications

- **Outlier Detection**: Identifying observations that deviate significantly from the distribution
- **Clustering**: Measuring distances in multivariate Gaussian mixture models
- **Classification**: Computing distances in discriminant analysis and Mahalanobis distance classifiers
- **Data Quality**: Detecting anomalies and unusual patterns in multivariate datasets

### Computational Steps

1. Compute the deviation vector: $d = x - \mu$
2. Compute the inverse of the covariance matrix: $\Sigma^{-1}$
3. Compute the quadratic form: $d^T \Sigma^{-1} d$
4. Take the square root to obtain the distance: $\sqrt{d^T \Sigma^{-1} d}$

### Example

For a 2D distribution with mean $\mu = [0, 0]$ and covariance matrix $\Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ (identity matrix), a point $x = [3, 4]$ has Mahalanobis distance:

$$D_M = \sqrt{[3, 4] \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} [3, 4]^T} = \sqrt{9 + 16} = 5$$

This is equivalent to the Euclidean distance since the covariance is identity.
