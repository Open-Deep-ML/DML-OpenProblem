## Understanding the SoftPlus Activation Function
Softplus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive.

### Mathematical Definition
The SoftPlus function is mathematically defined as:
     $$
     Softplus(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))
     $$
Where:
* $x$ : the input to the function
* $\beta$ : a positive scaling parameter that controls the sharpness of the function
* $threshold$ : a numerical stability threshold to prevent overflow if $\beta * x > threshold$