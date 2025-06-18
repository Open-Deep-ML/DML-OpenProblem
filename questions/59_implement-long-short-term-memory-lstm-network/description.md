## Task: Implement Long Short-Term Memory (LSTM) Network

Your task is to implement an LSTM network that processes a sequence of inputs and produces the final hidden state and cell state after processing all inputs.

Write a class `LSTM` with the following methods:

- `__init__(self, input_size, hidden_size)`: Initializes the LSTM with random weights and zero biases.
- `forward(self, x, initial_hidden_state, initial_cell_state)`: Processes a sequence of inputs and returns the hidden states at each time step, as well as the final hidden state and cell state.

The LSTM should compute the forget gate, input gate, candidate cell state, and output gate at each time step to update the hidden state and cell state.

    
