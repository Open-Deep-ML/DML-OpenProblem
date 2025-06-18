## Using Gini Impurity to Measure Disorder

One valid approach to measure disorder in a basket of apples is to use the **Gini impurity** metric. The Gini impurity is defined as:

$$
G = 1 - \sum_{i=1}^{k} p_i^2
$$

where:
- $p_i$ is the proportion of apples of the $i$-th color.
- $k$ is the total number of distinct colors.

### Key Properties

- **Single Color Case:** If all apples in the basket have the same color, then $p = 1$ and the Gini impurity is:
  $$
  G = 1 - 1^2 = 0
  $$
- **Increasing Disorder:** As the variety of colors increases, the impurity increases. For example:
  - Two equally frequent colors:  
    $$
    G = 1 - \left(0.5^2 + 0.5^2\right) = 0.5
    $$
  - Four equally frequent colors:  
    $$
    G = 1 - \left(4 \times 0.25^2\right) = 0.75
    $$

### Comparing Different Baskets

1. **Basket:** `[0,0,0,0]`  
   - Only one color -> $G = 0$
2. **Basket:** `[1,1,0,0]`  
   - Two colors, equal frequency -> $G = 0.5$
3. **Basket:** `[0,1,2,3]`  
   - Four equally frequent colors -> $G = 0.75$
4. **Basket:** `[0,0,1,1,2,2,3,3]`  
   - Equal distribution among four colors -> $G = 0.75$
5. **Basket:** `[0,0,0,0,0,1,2,3]`  
   - One dominant color, three others -> $G = 0.5625$

### Flexibility

While the Gini impurity is a suitable measure of disorder, any method that satisfies the following constraints is valid:
1. A basket with a single color must return a disorder score of **0**.
2. Baskets with more distinct colors must yield **higher disorder** scores.
3. The specific ordering constraints provided in the problem must be **maintained**.

By using this impurity measure, we can quantify how diverse a basket of apples is based on color distribution.
