
## Understanding Gradient Descent Variants with MSE Loss

Gradient Descent is an optimization algorithm used to minimize the cost function in machine learning models, particularly in linear regression and neural networks. The Mean Squared Error (MSE) loss function is commonly used in regression tasks. There are three main types of gradient descent based on how much data is used to compute the gradient at each iteration:

1. **Batch Gradient Descent**:  
   Batch Gradient Descent computes the gradient of the MSE loss function with respect to the parameters for the entire training dataset. It updates the parameters after processing the entire dataset:
   $$
   \theta = \theta - \alpha \cdot \frac{2}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
   $$
   where $ \alpha $ is the learning rate, $ m $ is the number of samples, and $ \nabla_{\theta} J(\theta) $ is the gradient of the MSE loss function.

2. **Stochastic Gradient Descent (SGD)**:  
   Stochastic Gradient Descent updates the parameters for each training example individually, making it faster but more noisy:
   $$
   \theta = \theta - \alpha \cdot 2 \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
   $$
   where $ x^{(i)}, y^{(i)} $ are individual training examples.

3. **Mini-Batch Gradient Descent**:  
   Mini-Batch Gradient Descent is a compromise between Batch and Stochastic Gradient Descent. It updates the parameters after processing a small batch of training examples, without shuffling the data:
   $$
   \theta = \theta - \alpha \cdot \frac{2}{b} \sum_{i=1}^{b} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
   $$
   where $ b $ is the batch size, a subset of the training dataset.

Each method has its advantages: Batch Gradient Descent is more stable but slower, Stochastic Gradient Descent is faster but noisy, and Mini-Batch Gradient Descent strikes a balance between the two.
