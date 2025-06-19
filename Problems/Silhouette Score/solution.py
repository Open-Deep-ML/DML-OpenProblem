import numpy as np

def silhouette_score(X, labels):
    if len(X) != len(labels):
        raise ValueError("X and labels must have the same number of samples")
    
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1 or n_clusters >= n_samples:
        raise ValueError("Number of clusters must be between 2 and n_samples-1")
    
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        distances[i] = np.sum((X - X[i]) ** 2, axis=1)
    distances = np.sqrt(distances)
    
    a_values = np.zeros(n_samples)
    b_values = np.zeros(n_samples)
    
    for i in range(n_samples):
        current_label = labels[i]
        
        same_cluster_mask = (labels == current_label)
        same_cluster_mask[i] = False  # Exclude self
        
        if np.any(same_cluster_mask):
            a_values[i] = np.mean(distances[i, same_cluster_mask])
        else:
            a_values[i] = 0  # Cluster of size 1
        
        # Get mean distances to other clusters
        other_labels = unique_labels[unique_labels != current_label]
        mean_distances = []
        
        for label in other_labels:
            other_cluster_mask = (labels == label)
            if np.any(other_cluster_mask):
                mean_distances.append(np.mean(distances[i, other_cluster_mask]))
        
        if mean_distances:
            b_values[i] = np.min(mean_distances)
        else:
            b_values[i] = 0  # Only one cluster (shouldn't happen due to checks)
    
    silhouette_values = np.zeros(n_samples)
    for i in range(n_samples):
        if a_values[i] == 0 and b_values[i] == 0:
            silhouette_values[i] = 0
        else:
            silhouette_values[i] = (b_values[i] - a_values[i]) / max(a_values[i], b_values[i])
    
    return np.mean(silhouette_values)
