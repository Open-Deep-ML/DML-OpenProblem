Implement the Adam (Adaptive Moment Estimation) optimization algorithm in Python. Adam is an optimization algorithm that adapts the learning rate for each parameter. Your task is to write a function `adam_optimizer` that updates the parameters of a given function using the Adam algorithm.

The function should take the following parameters:

- `f`: The objective function to be optimized
- `grad`: A function that computes the gradient of `f`
- `x0`: Initial parameter values
- `learning_rate`: The step size (default: 0.001)
- `beta1`: Exponential decay rate for the first moment estimates (default: 0.9)
- `beta2`: Exponential decay rate for the second moment estimates (default: 0.999)
- `epsilon`: A small constant for numerical stability (default: 1e-8)
- `num_iterations`: Number of iterations to run the optimizer (default: 1000)

The function should return the optimized parameters.
