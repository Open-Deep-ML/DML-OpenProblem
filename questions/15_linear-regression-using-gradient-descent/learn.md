
## Linear Regression Using Gradient Descent

Linear regression can also be performed using a technique called gradient descent, where the coefficients (or weights) of the model are iteratively adjusted to minimize a cost function (usually mean squared error). This method is particularly useful when the number of features is too large for analytical solutions like the normal equation or when the feature matrix is not invertible.

The gradient descent algorithm updates the weights by moving in the direction of the negative gradient of the cost function with respect to the weights. The updates occur iteratively until the algorithm converges to a minimum of the cost function.

The update rule for each weight is given by:
$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)x_j^{(i)}
$$

### Explanation of Terms
1. \( \alpha \) is the learning rate.
2. \( m \) is the number of training examples.
3. \( h_{\theta}(x^{(i)}) \) is the hypothesis function at iteration \( i \).
4. \( x^{(i)} \) is the feature vector of the \( i^{\text{th}} \) training example.
5. \( y^{(i)} \) is the actual target value for the \( i^{\text{th}} \) training example.
6. \( x_j^{(i)} \) is the value of feature \( j \) for the \( i^{\text{th}} \) training example.

### Key Points
- **Learning Rate**: The choice of learning rate is crucial for the convergence and performance of gradient descent. 
  - A small learning rate may lead to slow convergence.
  - A large learning rate may cause overshooting and divergence.
- **Number of Iterations**: The number of iterations determines how long the algorithm runs before it converges or stops.

### Practical Implementation
Implementing gradient descent involves initializing the weights, computing the gradient of the cost function, and iteratively updating the weights according to the update rule.
