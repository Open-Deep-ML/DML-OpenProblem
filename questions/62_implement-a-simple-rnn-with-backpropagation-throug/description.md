## Task: Implement a Simple RNN with Backpropagation Through Time (BPTT)

Your task is to implement a simple Recurrent Neural Network (RNN) and backpropagation through time (BPTT) to learn from sequential data. The RNN will process input sequences, update hidden states, and perform backpropagation to adjust weights based on the error gradient.

Write a class `SimpleRNN` with the following methods:

- `__init__(self, input_size, hidden_size, output_size)`: Initializes the RNN with random weights and zero biases.
- `forward(self, x)`: Processes a sequence of inputs and returns the hidden states and output.
- `backward(self, x, y, learning_rate)`: Performs backpropagation through time (BPTT) to adjust the weights based on the loss.

In this task, the RNN will be trained on sequence prediction, where the network will learn to predict the next item in a sequence. You should use 1/2 * Mean Squared Error (MSE) as the loss function and make sure to aggregate the losses at each time step by summing.
