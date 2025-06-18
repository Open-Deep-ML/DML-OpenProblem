## Derivative of a Polynomial

A function's derivative is a way of quantifying the function's slope at a given point. It allows us to understand whether the function is increasing, decreasing or flat at specific input. 

Taking the derivative of a polynomial at a single point follows a straight-forward rule. This question will show the rule and the edge case you should be on the look-out for.

### Mathematical Definition

When calculating the slope of a function $f(x)$, we usually require two points $x_{1}$ and $x_{2}$ and use the following formula:

$$
\frac{f(x_{2}) - f(x_{1})}{x_{2} - x_{1}}
$$

A derivative generalizes that notion by calculating the slope of a function at a specific point.
A derivative of a function $f(x)$ is mathematically defined as:

$$
\frac{d f(x)}{d x} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

Where:
- $x$ is the input to the function
- $h$ is the "step", which is equivalent to the difference $x_{2} - x_{1}$ in the two-point slope-formula

Taking the limit as the step grows smaller and smaller, allow us to quantify the slope at a certain point, instead of having to consider two points as in other methods of finding the slope.

When taking the derivative of a polynomial function $x^{n}$, where $n \neq 0$, then the derivative is: $n x^{n-1}$. In the special case where $n = 0$ then the derivative is zero. This is because $x^{0} = 1$ if $x \neq 0$.

A positive derivative indicates that the function is increasing in that point, a negative derivative indicates that the function is decreasing at that point. A derivative equal to zero indicates that the function is flat, which could potentially indicate a function's minimum or maximum.
