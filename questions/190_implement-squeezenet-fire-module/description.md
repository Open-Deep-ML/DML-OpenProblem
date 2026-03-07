Implement the Fire Module from SqueezeNet, a highly parameter-efficient convolutional neural network architecture. The Fire Module consists of a "squeeze" layer (1x1 convolutions) that feeds into an "expand" layer (a mix of 1x1 and 3x3 convolutions).

Your task is to implement the forward pass of the Fire Module using NumPy. Given an input tensor and weights/biases for the squeeze and expand layers, compute the output.

**Functions required:**

1.  `squeeze`: Applies 1x1 convolution to reduce channels.
2.  `expand`: Applies 1x1 and 3x3 convolutions in parallel and concatenates results.
3.  `fire_module`: Combines squeeze and expand.

Assume stride=1 and padding='same' for 3x3 convolutions (so spatial dimensions are preserved).
