## Understanding the GELU Activation Function

The **GELU** (Gaussian Error Linear Unit) is an activation function that combines properties of **ReLU** and **tanh (or) sigmoid** but adds a probabilistic interpretation, making it smooth and differentiable.

### Mathematical Definition

The GELU activation is defined as:

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

where $\Phi(x)$ is the cumulative distribution function (CDF) of the standard normal distribution:

$$
\Phi(x) = \frac{1}{2} \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

A widely used **approximation** is:

$$
\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
$$

> In some cases, a **sigmoid-based formula** may be used in place of **tanh** to approximate the erf function. If you wish to dive deep, you may read it [here](https://datascience.stackexchange.com/questions/49522/what-is-gelu-activation).

### Characteristics

- **Smooth and Nonlinear**: Unlike ReLU, which is piecewise linear, GELU is smooth and differentiable everywhere.
- **Retains Small Inputs**: It does not zero out all negative values like ReLU, but scales them down, which can be beneficial for gradient flow.
- **Stochastic Interpretation**: Treats input as a random variable and gates it based on the likelihood it is positive.

### Use in Practice

GELU is the default activation function in **Transformer-based models** such as BERT and GPT due to its ability to better capture complex relationships in data.
