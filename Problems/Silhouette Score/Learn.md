**Silhouette Score** is a metric for evaluating cluster quality, measuring how well an object in a cluster is similar to an 
object in its own cluster compared to other clusters. 

### Properties
- The silhouette score ranges from −1 to +1
- Higher values indicate better clustering
- Can be calculated with various distance metrics (e.g., Euclidean, Manhattan)

## Formula

For a point i in cluster A:
- **Cohesion (a)**: Average distance to points in the same cluster
- **Separation (b)**: Average distance to points in the nearest different cluster
- **Silhouette Score s(i)**: `s(i) = (b - a) / max(a, b)`

- The cluster-level silhouette score is calculated by averaging the silhouette scores of all points in the cluster:

1. **Individual Point Calculation**: Compute s(i) for each point
2. **Cluster Score**: `SC = (1/n) * Σ s(i)` 
   Where:
   - n is the number of points in the cluster
   - Σ s(i) is the sum of individual point silhouette scores

### Overall Dataset Silhouette Score
For the entire dataset with k clusters:
- **Overall Score**: `S = (1/N) * Σ SC(j)`
   Where:
   - N is the total number of points
   - Σ SC(j) is the sum of individual cluster silhouette scores
   - k is the number of clusters

## Example Calculation
| Point | X | Cluster |
|-------|---|---------|
| A     | 1 | 0       |
| B     | 2 | 0       |
| C     | 8 | 1       |
| D     | 9 | 1       |

### Calculation for Point A
1. **Cohesion (a)**: Avg distance to same cluster  
   Distance(A→B) = |1-2| = 1  
   `a(A) = 1`
2. **Separation (b)**: Avg distance to nearest cluster  
   Distance(A→C) = 7, Distance(A→D) = 8  
   `b(A) = min(7,8) = 7`
3. **Silhouette Score**:  
   `s(A) = (b - a)/max(a,b) = (7-1)/7 ≈ 0.86`

### Aggregate Scores
| Point | s(i)  |
|-------|-------|
| A     | 0.86  |
| B     | 0.83  |
| C     | 0.83  |
| D     | 0.86  |

**Final Score**: `mean(s(i)) = 0.85` 
