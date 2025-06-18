import numpy as np
from tinygrad.tensor import Tensor

def k_means_clustering_tg(points, k, initial_centroids, max_iterations) -> list[tuple[float, ...]]:
    """
    Perform k-means clustering on `points` into `k` clusters using tinygrad.
    points: list of lists or Tensor, shape (n_points, n_features)
    initial_centroids: list of lists or Tensor, shape (k, n_features)
    max_iterations: maximum number of iterations
    Returns a list of k centroids as tuples, rounded to 4 decimals.
    """
    pts = np.array(points, dtype=float)
    centroids = np.array(initial_centroids, dtype=float)
    for _ in range(max_iterations):
        # compute distances (k, n_points)
        dists = np.array([np.linalg.norm(pts - c, axis=1) for c in centroids])
        # assign points
        assignments = dists.argmin(axis=0)
        new_centroids = np.array([
            pts[assignments == i].mean(axis=0) if np.any(assignments == i) else centroids[i]
            for i in range(k)
        ])
        new_centroids = np.round(new_centroids, 4)
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return [tuple(c.tolist()) for c in centroids]
