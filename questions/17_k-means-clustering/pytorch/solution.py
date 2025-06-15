import torch

def k_means_clustering(points, k, initial_centroids, max_iterations) -> list[tuple[float, ...]]:
    """
    Perform k-means clustering on `points` into `k` clusters.
    points: tensor of shape (n_points, n_features)
    initial_centroids: tensor of shape (k, n_features)
    max_iterations: maximum number of iterations
    Returns a list of k centroids as tuples, rounded to 4 decimals.
    """
    points_t = torch.as_tensor(points, dtype=torch.float)
    centroids = torch.as_tensor(initial_centroids, dtype=torch.float)
    for _ in range(max_iterations):
        # compute distances (k, n_points)
        diffs = points_t.unsqueeze(0) - centroids.unsqueeze(1)
        distances = torch.sqrt((diffs ** 2).sum(dim=2))
        # assign each point to nearest centroid
        assignments = distances.argmin(dim=0)
        new_centroids = []
        for i in range(k):
            cluster_points = points_t[assignments == i]
            if cluster_points.numel() == 0:
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(cluster_points.mean(dim=0))
        new_centroids = torch.stack(new_centroids)
        new_centroids = torch.round(new_centroids * 10000) / 10000
        if torch.equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return [tuple(c.tolist()) for c in centroids]
