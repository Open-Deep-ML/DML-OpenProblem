import numpy as np

def silhouette_score(X, labels):
    if len(X) != len(labels):
        raise ValueError("X and labels must have the same number of samples")
    
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1 or n_clusters >= n_samples:
        raise ValueError("Number of clusters must be between 2 and n_samples-1")
    
    # Initialize arrays for a and b values
    a_values = np.zeros(n_samples)
    b_values = np.zeros(n_samples)
    
    # Calculate a(i) and b(i) for each sample
    for i in range(n_samples):
        current_label = labels[i]
        
        # Compute distances to all points
        distances = np.zeros((n_samples,))
        for j in range(n_samples):
            if i != j:  # Skip the point itself
                distances[j] = np.sqrt(np.sum((X[i] - X[j])**2))
            else:
                distances[j] = np.inf  # Set self-distance to infinity
        
        # Calculate a(i): mean distance to points in the same cluster
        same_cluster_indices = np.where(labels == current_label)[0]
        same_cluster_indices = same_cluster_indices[same_cluster_indices != i]  # Remove self
        
        if len(same_cluster_indices) > 0:
            a_values[i] = np.mean(distances[same_cluster_indices])
        else:
            a_values[i] = 0
        
        # Calculate b(i): mean distance to points in the nearest cluster
        min_mean_distance = np.inf
        for label in unique_labels:
            if label != current_label:
                other_cluster_indices = np.where(labels == label)[0]
                if len(other_cluster_indices) > 0:
                    mean_distance = np.mean(distances[other_cluster_indices])
                    if mean_distance < min_mean_distance:
                        min_mean_distance = mean_distance
        
        b_values[i] = min_mean_distance
    
    # Calculate silhouette values for each sample
    silhouette_values = np.zeros(n_samples)
    for i in range(n_samples):
        if a_values[i] == 0 and b_values[i] == 0:
            silhouette_values[i] = 0
        elif b_values[i] == np.inf:
            silhouette_values[i] = 0
        else:
            silhouette_values[i] = (b_values[i] - a_values[i]) / max(a_values[i], b_values[i])
    
    # Return the mean silhouette value
    return np.mean(silhouette_values)


def test_silhouette_score():
    # Test Case 1: Well-separated clusters
    X1 = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],  # Cluster 0
        [5, 5], [6, 5], [5, 6], [6, 6],  # Cluster 1
        [0, 5], [1, 5], [0, 6], [1, 6]   # Cluster 2
    ])
    labels1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    score1 = silhouette_score(X1, labels1)
    print(f"Test Case 1 Score: {score1:.3f}")
    assert score1 > 0.7, "Well-separated clusters should have high silhouette score"
    
    # Test Case 2: Overlapping clusters
    X2 = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],       # Cluster 0
        [0.5, 0.5], [1.5, 0.5], [0.5, 1.5],   # Cluster 1
        [3, 3], [4, 3], [3, 4], [4, 4]        # Cluster 2
    ])
    labels2 = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    score2 = silhouette_score(X2, labels2)
    print(f"Test Case 2 Score: {score2:.3f}")
    assert score2 < score1, "Overlapping clusters should have lower score than well-separated clusters"
    
    # Test Case 3: Bad clustering
    X3 = np.array([
        [1, 1], [1, 2], [2, 1], [2, 2],
        [1, 3], [2, 3], [3, 1], [3, 2]
    ])
    labels3 = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # Not meaningful clusters
    score3 = silhouette_score(X3, labels3)
    print(f"Test Case 3 Score: {score3:.3f}")
    assert score3 < 0.5, "Bad clustering should have low silhouette score"
    
    # Test Case 4: Error cases
    try:
        silhouette_score(X1, np.ones(len(X1)))
        assert False, "Single cluster should raise ValueError"
    except ValueError:
        pass
    
    try:
        silhouette_score(X1, np.arange(len(X1)))
        assert False, "Too many clusters should raise ValueError"
    except ValueError:
        pass
    
    try:
        silhouette_score(X1, labels1[:5])
        assert False, "Mismatched X and labels should raise ValueError"
    except ValueError:
        pass
    
    print("All Test Cases Passed!")

if __name__ == "__main__":
    test_silhouette_score()
