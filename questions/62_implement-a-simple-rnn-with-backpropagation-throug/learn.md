# Understanding Recurrent Neural Networks (RNNs) and Backpropagation Through Time (BPTT)

Recurrent Neural Networks (RNNs) are designed to handle sequential data by maintaining a hidden state that captures information from previous inputs. They are particularly useful for tasks where context or sequential order is important, such as language modeling, time series forecasting, and sequence prediction.

## RNN Architecture

An RNN processes inputs one at a time while maintaining a hidden state that gets updated at each time step. The core equations governing the forward pass of an RNN are:

### 1) Hidden State Update

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

### 2) Output Computation

$$
y_t = W_{hy} h_t + b_y
$$

Where:

1. $x_t$ is the input at time step $t$.
2. $h_t$ is the hidden state at time step $t$.
3. $W_{xh}$ is the weight matrix for input to hidden state.
4. $W_{hh}$ is the weight matrix for hidden state to hidden state.
5. $W_{hy}$ is the weight matrix for hidden state to output.
6. $b_h$ and $b_y$ are the bias terms for the hidden state and output, respectively.
7. $\tanh$ is the hyperbolic tangent activation function applied element-wise.

## Forward Pass Implementation

In the forward pass, we iterate over each element in the input sequence, updating the hidden state and computing the output:

1. **Initialize the hidden state** $h_0$ to zeros.
2. For each time step $t$:
   - Compute $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$.
   - Compute $y_t = W_{hy} h_t + b_y$.
   - Store $h_t$ and $y_t$ for use in backpropagation.

## Loss Function

The loss function measures the discrepancy between the predicted outputs and the actual target values. For sequence prediction tasks, we often use the **Mean Squared Error (MSE)** loss:

$$
\text{Loss} = \frac{1}{T} \sum_{t=1}^{T} (\hat{y}_t - y_t)^2
$$

Where $T$ is the length of the sequence, $\hat{y}_t$ is the predicted output, and $y_t$ is the actual target at time step $t$.

## Backpropagation Through Time (BPTT)

BPTT is the process of training RNNs by unrolling them through time and applying backpropagation to compute gradients for each time step. The key steps in BPTT are:

1. Compute the gradient of the loss with respect to the outputs:

$$
\frac{dL}{dy_t} = \hat{y}_t - y_t
$$

2. Compute the gradients for the output layer weights and biases:

$$
dW_{hy} += \frac{dL}{dy_t} \cdot h_t^T
$$

$$
db_y += \frac{dL}{dy_t}
$$

3. Backpropagate the gradients through the hidden layers:

$$
dh_t = W_{hy}^T \cdot \frac{dL}{dy_t} + dh_{t+1}
$$

$$
dh_{\text{raw}} = dh_t \circ (1 - h_t^2)
$$

Here, $\circ$ denotes element-wise multiplication, and $(1 - h_t^2)$ is the derivative of the $\tanh$ activation function.

4. Compute the gradients for the hidden layer weights and biases:

$$
dW_{xh} += dh_{\text{raw}} \cdot x_t^T
$$

$$
dW_{hh} += dh_{\text{raw}} \cdot h_{t-1}^T
$$

$$
db_h += dh_{\text{raw}}
$$

We repeat steps 1-4 for each time step $t$ in reverse order (from $T$ to 1), accumulating the gradients. The term $dh_{t+1}$ represents the gradient flowing from the next time step, initialized to zeros at the last time step.

## Updating Weights

After computing the gradients, we update the weights and biases using gradient descent:

$$
W_{xh} -= \text{learning\_rate} \times dW_{xh}
$$

$$
W_{hh} -= \text{learning\_rate} \times dW_{hh}
$$

$$
W_{hy} -= \text{learning\_rate} \times dW_{hy}
$$

$$
b_h -= \text{learning\_rate} \times db_h
$$

$$
b_y -= \text{learning\_rate} \times db_y
$$

## Implementing the RNN

To implement the RNN with BPTT, follow these steps:

1. **Initialization**: Initialize the weight matrices $W_{xh}, W_{hh}, W_{hy}$ with small random values and biases $b_h, b_y$ with zeros.
2. **Forward Pass**: Implement the forward method to process the input sequence, updating the hidden states and computing the outputs at each time step. Store the inputs, hidden states, and outputs for use in backpropagation.
3. **Backward Pass**: Implement the backward method to perform BPTT. Compute the gradients at each time step in reverse order, accumulate them, and update the weights and biases.
4. **Training Loop**: Train the RNN over multiple epochs by repeatedly performing forward and backward passes and updating the weights.

## Tips for Implementation

1. **Gradient Clipping**: To prevent exploding gradients, consider applying gradient clipping, which scales down gradients if they exceed a certain threshold.
2. **Learning Rate**: Choose an appropriate learning rate. If the learning rate is too high, the training may become unstable.
3. **Debugging**: Check the dimensions of all matrices and vectors to ensure they align correctly during matrix multiplication.
4. **Testing**: Start with small sequences and hidden sizes to test your implementation before scaling up.

## Example Calculation

Suppose we have an input sequence $x = [x_1, x_2]$ and target sequence $y = [y_1, y_2]$. Here's how you would compute the forward and backward passes:

### 1) Forward Pass

- At $t = 1$:
  - Compute $h_1 = \tanh(W_{xh} x_1 + W_{hh} h_0 + b_h)$.
  - Compute $\hat{y}_1 = W_{hy} h_1 + b_y$.

- At $t = 2$:
  - Compute $h_2 = \tanh(W_{xh} x_2 + W_{hh} h_1 + b_h)$.
  - Compute $\hat{y}_2 = W_{hy} h_2 + b_y$.

### 2) Compute Loss

$$
L = \frac{1}{2} \left[ (\hat{y}_1 - y_1)^2 + (\hat{y}_2 - y_2)^2 \right]
$$

### 3) Backward Pass

Starting from $t = 2$ to $t = 1$:

- At $t = 2$:
  - Compute $\frac{dL}{d\hat{y}_2} = \hat{y}_2 - y_2$.
  - Backpropagate to find $dh_2$, $dW_{hy}$, $db_y$.

- At $t = 1$:
  - Use the chain rule to compute gradients with respect to inputs and weights.

## Conclusion

RNNs with BPTT are powerful for modeling sequences, but they come with challenges such as vanishing and exploding gradients. Techniques like gradient clipping, long short-term memory (LSTM) cells, or gated recurrent units (GRUs) can help mitigate these issues. Implementing an RNN from scratch provides a deeper understanding of the underlying mechanisms and prepares you for working with more complex architectures.
