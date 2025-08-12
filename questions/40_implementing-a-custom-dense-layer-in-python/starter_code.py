import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
    
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
	def __init__(self, n_units, input_shape=None):
		self.layer_input = None
		self.input_shape = input_shape
		self.n_units = n_units
		self.trainable = True
		self.W = None
		self.w0 = None

	def forward_pass(self):
        # Your code here
		pass


	def backward_pass(self, accum_grad):
        # Your code here
		pass


	def number_of_parameters(self):
        # Your code here
		pass
