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

if __name__ == "__main__":
    test_k_means_clustering()
    print("All k_means_clustering tests passed.")
