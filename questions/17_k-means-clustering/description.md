Your task is to write a Python function that implements the k-Means clustering algorithm. This function should take specific inputs and produce a list of final centroids. k-Means clustering is a method used to partition `n` points into `k` clusters. The goal is to group similar points together and represent each group by its center (called the *centroid*).

### Function Inputs:

- `points`: A list of points, where each point is a tuple of coordinates (e.g., `(x, y)` for 2D points)
- `k`: An integer representing the number of clusters to form
- `initial_centroids`: A list of initial centroid points, each a tuple of coordinates
- `max_iterations`: An integer representing the maximum number of iterations to perform

### Function Output:

A list of the final centroids of the clusters, where each centroid is rounded to the nearest fourth decimal.
