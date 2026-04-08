Write a Python function that calculates the Mahalanobis distance between a point and a probability distribution. The function should take a point, the mean vector of the distribution, and the covariance matrix as inputs and return the Mahalanobis distance as a scalar value. The Mahalanobis distance is a measure of how many standard deviations away a point is from the mean, considering the correlations in the data (represented by the covariance matrix).

The function inputs should be in NumPy format (numpy.ndarray):
- `point`: A 1D NumPy array representing the point in the feature space
- `mean`: A 1D NumPy array representing the mean of the distribution
- `cov_matrix`: A 2D NumPy array representing the covariance matrix of the distribution

The function should return a float representing the Mahalanobis distance computed as: $D_M = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$

**Note:** The return value should be rounded to 4 decimal places.
