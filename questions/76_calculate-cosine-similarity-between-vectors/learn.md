
## Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors. It doesn't consider the magnitude of the vectors but focuses on the angle between them.

### Cosine Similarity Formula
$$
\cos(\theta) = \frac{\sum_{i=1}^{p} A_i B_i}{\sqrt{\sum_{i=1}^{p} A_i^2} \sqrt{\sum_{i=1}^{p} B_i^2}}
$$

### Implementation Steps for Cosine Similarity
1. **Handle Input**: Ensure input vectors have the same dimensions and handle edge cases (e.g., zero vectors).
2. **Dot Product**: Compute $\sum_{i=1}^{p} A_i B_i $ for the two vectors.
3. **Magnitudes**: Compute the L2 norms $ \sqrt{\sum_{i=1}^{p} A_i^2} $ and $ \sqrt{\sum_{i=1}^{p} B_i^2} $.
4. **Final Result**: Divide the dot product by the product of the magnitudes.

### Use Cases
1. **Text and Image Similarity**
2. **Recommendation Systems**
3. **Query Matching**

### Pitfalls
1. **Magnitude Blindness**:
   - Example:
     - $ \text{vector1} = (1, 1) $
     - $ \text{vector2} = (1000, 1000) $
     - Cosine similarity $ = 1 $, despite the vastly different magnitudes.
2. **Sparse Data Issues**:
   - In high-dimensional spaces, where data is often sparse, cosine similarity may become less reliable.
3. **Non-Negative Data Limitation**:
   - If all values are positive, cosine similarity cannot capture negative relationships or inverse trends.
