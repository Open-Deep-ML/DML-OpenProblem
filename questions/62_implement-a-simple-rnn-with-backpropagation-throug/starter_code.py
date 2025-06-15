import numpy as np
class SimpleRNN:
	def __init__(self, input_size, hidden_size, output_size):
		"""
		Initializes the RNN with random weights and zero biases.
		"""
		self.hidden_size = hidden_size
		self.W_xh = np.random.randn(hidden_size, input_size)*0.01
		self.W_hh = np.random.randn(hidden_size, hidden_size)*0.01
		self.W_hy = np.random.randn(output_size, hidden_size)*0.01
		self.b_h = np.zeros((hidden_size, 1))
		self.b_y = np.zeros((output_size, 1))
	def forward(self, x):
		"""
		Forward pass through the RNN for a given sequence of inputs.
		"""
		pass

	def backward(self, x, y, learning_rate):
		"""
		Backpropagation through time to adjust weights based on error gradient.
		"""
		pass
