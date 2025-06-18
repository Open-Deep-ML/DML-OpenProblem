
## Understanding Long Short-Term Memory Networks (LSTMs)

Long Short-Term Memory Networks are a special type of RNN designed to capture long-term dependencies in sequential data by using a more complex hidden state structure.

### LSTM Gates and Their Functions

For each time step $t$, the LSTM updates its cell state $c_t$ and hidden state $h_t$ using the current input $x_t$, the previous cell state $c_{t-1}$, and the previous hidden state $h_{t-1}$. The LSTM architecture consists of several gates that control the flow of information:

#### Forget Gate $f_t$:

This gate decides what information to discard from the cell state. It looks at the previous hidden state $h_{t-1}$ and the current input $x_t$, and outputs a number between 0 and 1 for each number in the cell state. A 1 represents "keep this" while a 0 represents "forget this".

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

#### Input Gate $i_t$:

This gate decides which new information will be stored in the cell state. It consists of two parts:
- A sigmoid layer that decides which values we'll update.
- A tanh layer that creates a vector of new candidate values $\tilde{c}_t$ that could be added to the state.

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
$$

#### Cell State Update $c_t$:

This step updates the old cell state $c_{t-1}$ into the new cell state $c_t$. It multiplies the old state by the forget gate output, then adds the product of the input gate and the new candidate values.

$$
c_t = f_t \circ c_{t-1} + i_t \circ \tilde{c}_t
$$

#### Output Gate $o_t$:

This gate decides what parts of the cell state we're going to output. It uses a sigmoid function to determine which parts of the cell state to output, and then multiplies it by a tanh of the cell state to get the final output.

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \circ \tanh(c_t)
$$

Where:
- $(W_f, W_i, W_c, W_o)$ are weight matrices for the forget gate, input gate, cell state, and output gate respectively.
- $(b_f, b_i, b_c, b_o)$ are bias vectors.
- $\sigma$ is the sigmoid activation function.
- $\circ$ denotes element-wise multiplication.

### Implementation Steps

1. **Initialization**: Start with the initial cell state $c_0$ and hidden state $h_0$.
2. **Sequence Processing**: For each input $x_t$ in the sequence:
   - Compute forget gate $f_t$, input gate $i_t$, candidate cell state $\tilde{c}_t$, and output gate $o_t$.
   - Update cell state $c_t$ and hidden state $h_t$.
3. **Final Output**: After processing all inputs, the final hidden state $h_T$ (where $T$ is the length of the sequence) contains information from the entire sequence.

### Example Calculation

Given:
- Inputs: $x_1 = 1.0$, $x_2 = 2.0$, $x_3 = 3.0$
- Initial states: $c_0 = 0.0$, $h_0 = 0.0$
- Simplified weights (for demonstration): $W_f = W_i = W_c = W_o = 0.5$
- All biases: $b_f = b_i = b_c = b_o = 0.1$

#### Compute:

**First time step $t = 1$:**

$$
f_1 = \sigma(0.5 \times 1.0 + 0.1) = 0.6487
$$

$$
i_1 = \sigma(0.5 \times 1.0 + 0.1) = 0.6487
$$

$$
\tilde{c}_1 = \tanh(0.5 \times 1.0 + 0.1) = 0.5370
$$

$$
c_1 = f_1 \times 0.0 + i_1 \times \tilde{c}_1 = 0.6487 \times 0.0 + 0.6487 \times 0.5370 = 0.3484
$$

$$
o_1 = \sigma(0.5 \times 1.0 + 0.1) = 0.6487
$$

$$
h_1 = o_1 \times \tanh(c_1) = 0.6487 \times \tanh(0.3484) = 0.2169
$$

**Second time step $t = 2$:**
(Calculations omitted for brevity, but follow the same pattern using $x_2 = 2.0$ and the previous states)

**Third time step $t = 3$:**
(Calculations omitted for brevity, but follow the same pattern using $x_3 = 3.0$ and the previous states)

The final hidden state $h_3$ would be the result after these calculations.

### Applications

LSTMs are extensively used in various sequence modeling tasks, including machine translation, speech recognition, and time series forecasting, where capturing long-term dependencies is crucial.
