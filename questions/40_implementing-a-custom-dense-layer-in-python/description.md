## Implementing a Custom Dense Layer in Python

You are provided with a base `Layer` class that defines the structure of a neural network layer. Your task is to implement a subclass called `Dense`, which represents a fully connected neural network layer. The `Dense` class should extend the `Layer` class and implement the following methods:

1. **Initialization (`__init__`)**:
   - Define the layer with a specified number of neurons (`n_units`) and an optional input shape (`input_shape`).
   - Set up placeholders for the layer's weights (`W`), biases (`w0`), and optimizers.

2. **Weight Initialization (`initialize`)**:
   - Initialize the weights `W` using a uniform distribution with a limit of `1 / sqrt(input_shape[0])`, and bias `w0` should be set to zero.
   - Initialize optimizers for `W` and `w0`.

3. **Parameter Count (`parameters`)**:
   - Return the total number of trainable parameters in the layer, which includes the parameters in `W` and `w0`.

4. **Forward Pass (`forward_pass`)**:
   - Compute the output of the layer by performing a dot product between the input `X` and the weight matrix `W`, and then adding the bias `w0`.

5. **Backward Pass (`backward_pass`)**:
   - Calculate and return the gradient with respect to the input.
   - If the layer is trainable, update the weights and biases using the optimizer's update rule.

6. **Output Shape (`output_shape`)**:
   - Return the shape of the output produced by the forward pass, which should be `(self.n_units,)`.

**Objective**:  
Extend the `Layer` class by implementing the `Dense` class to ensure it functions correctly within a neural network framework.
