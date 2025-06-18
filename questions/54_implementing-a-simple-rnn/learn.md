
## Understanding Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a class of neural networks designed to handle sequential data by maintaining a hidden state that captures information from previous inputs.

### Mathematical Formulation

For each time step $t$, the RNN updates its hidden state $h_t$ using the current input $x_t$ and the previous hidden state $h_{t-1}$:

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

Where:
1. $W_x$ is the weight matrix for the input-to-hidden connections.
2. $W_h$ is the weight matrix for the hidden-to-hidden connections.
3. $b$ is the bias vector.
4. $\tanh$ is the hyperbolic tangent activation function applied element-wise.

### Implementation Steps

1. **Initialization**: Start with the initial hidden state $h_0$.

2. **Sequence Processing**: For each input $x_t$ in the sequence:

   $$
   h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
   $$

3. **Final Output**: After processing all inputs, the final hidden state $h_T$ (where $T$ is the length of the sequence) contains information from the entire sequence.

### Example Calculation

Given:
1. Inputs: $x_1 = 1.0$, $x_2 = 2.0$, $x_3 = 3.0$
2. Initial hidden state: $h_0 = 0.0$
3. Weights:
   - $W_x = 0.5$
   - $W_h = 0.8$
4. Bias: $b = 0.0$

**Compute**:

1. First time step ($t = 1$):

   $$
   h_1 = \tanh(0.5 \times 1.0 + 0.8 \times 0.0 + 0.0) = \tanh(0.5) \approx 0.4621
   $$

2. Second time step ($t = 2$):

   $$
   h_2 = \tanh(0.5 \times 2.0 + 0.8 \times 0.4621 + 0.0) = \tanh(1.0 + 0.3697) = \tanh(1.3697) \approx 0.8781
   $$

3. Third time step ($t = 3$):

   $$
   h_3 = \tanh(0.5 \times 3.0 + 0.8 \times 0.8781 + 0.0) = \tanh(1.5 + 0.7025) = \tanh(2.2025) \approx 0.9750
   $$

The final hidden state $h_3$ is approximately 0.9750.

### Applications

RNNs are widely used in natural language processing, time-series prediction, and any task involving sequential data.
