# Example: Mahalanobis Distance

## Example Input

```python
point = np.array([3.0, 4.0, 2.0])
mean = np.array([0.0, 0.0, 0.0])
cov_matrix = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
```

## Example Output

```
5.744562646538029
```

## Reasoning

With an identity covariance matrix, the Mahalanobis distance reduces to the Euclidean distance. The point [3, 4, 2] is at distance sqrt(3^2 + 4^2 + 2^2) = sqrt(9 + 16 + 4) = sqrt(29) ≈ 5.745 from the origin [0, 0, 0]. The covariance matrix being identity means there is no correlation between dimensions and all dimensions have unit variance, so the standardized distance equals the Euclidean distance.
