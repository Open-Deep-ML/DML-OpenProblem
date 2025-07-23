def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
    """
    if not points or k <= 0:
        return []
    
    if k > len(points):
        k = len(points)
    
    # Convert to numpy arrays for vectorized operations
    points_array = np.array(points)
    query_array = np.array(query_point)
    
    # Calculate Euclidean distances using broadcasting
    distances = np.sqrt(np.sum((points_array - query_array) ** 2, axis=1))
    
    # Get indices of k smallest distances
    k_nearest_indices = np.argsort(distances)[:k]
    
    # Return the k nearest points as tuples
    return [tuple(points_array[i]) for i in k_nearest_indices]