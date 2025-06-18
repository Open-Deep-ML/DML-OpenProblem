import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))

def k_means_clustering(points, k, initial_centroids, max_iterations):
    points = np.array(points)
    centroids = np.array(initial_centroids)
    
    for iteration in range(max_iterations):
        # Assign points to the nearest centroid
        distances = np.array([euclidean_distance(points, centroid) for centroid in centroids])
        assignments = np.argmin(distances, axis=0)

        new_centroids = np.array([points[assignments == i].mean(axis=0) if len(points[assignments == i]) > 0 else centroids[i] for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        centroids = np.round(centroids, 4)
    
    return [tuple(centroid) for centroid in centroids]

def test_k_means_clustering() -> None:
    # Test case 1
    points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
    k = 2
    initial_centroids = [(1, 1), (10, 1)]
    max_iterations = 10
    assert k_means_clustering(points, k, initial_centroids, max_iterations) == [(1.0, 2.0), (10.0, 2.0)], "Test case 1 failed"

    # Test case 2
    points = [(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)]
    k = 2
    initial_centroids = [(1, 1, 1), (10, 10, 10)]
    max_iterations = 10
    assert k_means_clustering(points, k, initial_centroids, max_iterations) == [(1.0, 1.0, 1.0), (10.3333, 10.6667, 10.3333)], "Test case 2 failed"

    # Test case 3: Single cluster
    points = [(1, 1), (2, 2), (3, 3), (4, 4)]
    k = 1
    initial_centroids = [(0, 0)]
    max_iterations = 10
    assert k_means_clustering(points, k, initial_centroids, max_iterations) == [(2.5, 2.5)], "Test case 3 failed"

    # Test case 4: Four clusters in 2D space
    points = [(0, 0), (1, 0), (0, 1), (1, 1), (5, 5), (6, 5), (5, 6), (6, 6),
              (0, 5), (1, 5), (0, 6), (1, 6), (5, 0), (6, 0), (5, 1), (6, 1)]
    k = 4
    initial_centroids = [(0, 0), (0, 5), (5, 0), (5, 5)]
    max_iterations = 10
    result = k_means_clustering(points, k, initial_centroids, max_iterations)
    expected = [(0.5, 0.5), (0.5, 5.5), (5.5, 0.5), (5.5, 5.5)]
    assert all(np.allclose(r, e) for r, e in zip(result, expected)), "Test case 4 failed"

    # Test case 5: Clusters with different densities
    points = [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5),  # Dense cluster
              (4, 4), (6, 6)]  # Sparse cluster
    k = 2
    initial_centroids = [(0, 0), (5, 5)]
    max_iterations = 10
    result = k_means_clustering(points, k, initial_centroids, max_iterations)
    expected = [(0.25, 0.25), (5.0, 5.0)]
    assert all(np.allclose(r, e) for r, e in zip(result, expected)), "Test case 5 failed"

if __name__ == "__main__":
    test_k_means_clustering()
    print("All k_means_clustering tests passed.")
    