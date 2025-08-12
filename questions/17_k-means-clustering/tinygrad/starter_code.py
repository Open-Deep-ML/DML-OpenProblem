def k_means_clustering_tg(
    points, k, initial_centroids, max_iterations
) -> list[tuple[float, ...]]:
    """
    Perform k-means clustering on `points` into `k` clusters using tinygrad.
    points: list of lists or Tensor, shape (n_points, n_features)
    initial_centroids: list of lists or Tensor, shape (k, n_features)
    max_iterations: maximum number of iterations
    Returns a list of k centroids as tuples, rounded to 4 decimals.
    """
    # Your implementation here
    pass
