## Understanding the Dense Layer

The Dense layer, also known as a fully connected layer, is a fundamental building block in neural networks. It connects each input neuron to each output neuron, hence the term "fully connected."

### 1. Weight Initialization
In the `initialize` method, weights are typically initialized using a uniform distribution within a certain range. For a Dense layer, a common practice is to set this range as:
$$
\text{limit} = \frac{1}{\sqrt{\text{input\_shape}}}
$$
This initialization helps in maintaining a balance in the distribution of weights, preventing issues like vanishing or exploding gradients during training.

### 2. Forward Pass
During the forward pass, the input data \( X \) is multiplied by the weight matrix \( W \) and added to the bias \( w_0 \) to produce the output:
$$
\text{output} = X \cdot W + w_0
$$

### 3. Backward Pass
The backward pass computes the gradients of the loss function with respect to the input data, weights, and biases. If the layer is trainable, it updates the weights and biases using the optimizer's update rule:
$$
W = W - \eta \cdot \text{grad}_W
$$
$$
w_0 = w_0 - \eta \cdot \text{grad}_{w_0}
$$
where \( \eta \) is the learning rate and \( \text{grad}_W \) and \( \text{grad}_{w_0} \) are the gradients of the weights and biases, respectively.

### 4. Output Shape
The shape of the output from a Dense layer is determined by the number of neurons in the layer. If a layer has `n_units` neurons, the output shape will be \( (n\_units,) \).

### Resources
- [CS231n: Fully Connected Layer](https://cs231n.github.io/neural-networks-2/#fc)

    
