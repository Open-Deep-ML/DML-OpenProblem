
## K-Means Clustering Algorithm Implementation

### Algorithm Steps

1. **Initialization**  
   Use the provided `initial_centroids` as your starting point. This step is already done for you in the input.

2. **Assignment Step**  
   For each point in your dataset:
   - Calculate its distance to each centroid.
   - Assign the point to the cluster of the nearest centroid.  
   *Hint*: Consider creating a helper function to calculate the Euclidean distance between two points.

3. **Update Step**  
   For each cluster:
   - Calculate the mean of all points assigned to the cluster.
   - Update the centroid to this new mean position.  
   *Hint*: Be careful with potential empty clusters. Decide how you'll handle them (e.g., keep the previous centroid).

4. **Iteration**  
   Repeat steps 2 and 3 until either:
   - The centroids no longer change significantly (this case does not need to be included in your solution), or
   - You reach the `max_iterations` limit.  
   *Hint*: You might want to keep track of the previous centroids to check for significant changes.

5. **Result**  
   Return the list of final centroids, ensuring each coordinate is rounded to the nearest fourth decimal.
