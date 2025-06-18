
## Neural Network Learning with Backpropagation

This task involves implementing backpropagation for a single neuron in a neural network. The neuron processes inputs and updates parameters to minimize the Mean Squared Error (MSE) between predicted outputs and true labels.

### Mathematical Background

**Forward Pass**  
Compute the neuron output by calculating the dot product of the weights and input features, and adding the bias:
$$
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Loss Calculation (MSE)**  
The Mean Squared Error quantifies the error between the neuron's predictions and the actual labels:
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i)^2
$$

### Backward Pass (Gradient Calculation)
Compute the gradient of the MSE with respect to each weight and the bias. This involves the partial derivatives of the loss function with respect to the output of the neuron, multiplied by the derivative of the sigmoid function:
$$
\frac{\partial MSE}{\partial w_j} = \frac{2}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i) \sigma'(z_i) x_{ij}
$$
$$
\frac{\partial MSE}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i) \sigma'(z_i)
$$

### Parameter Update
Update each weight and the bias by subtracting a portion of the gradient, determined by the learning rate:
$$
w_j = w_j - \alpha \frac{\partial MSE}{\partial w_j}
$$
$$
b = b - \alpha \frac{\partial MSE}{\partial b}
$$

### Practical Implementation
This process refines the neuron's ability to predict accurately by iteratively adjusting the weights and bias based on the error gradients, optimizing the neural network's performance over multiple iterations.
