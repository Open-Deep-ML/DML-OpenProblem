## How to Find the Minimum of a Function

To find the minimum of a function like

$$
f(x) = x^4 - 3x^3 + 2
$$

we can use a technique called **gradient descent**.

### Steps:

1. **Find the Derivative**
   - The derivative (slope) tells us which direction the function is increasing or decreasing.
   - For this problem, the derivative is:
     $$
     f'(x) = 4x^3 - 9x^2
     $$

2. **Move Opposite the Slope**
   - If the slope is positive, move left.
   - If the slope is negative, move right.
   - Update the position by:
     $$
     x_{new} = x_{old} - \text{learning rate} \times f'(x_{old})
     $$

3. **Repeat**
   - Keep updating $x$ until the change is very small (below a tolerance).

### Why Does This Work?
- If you always move downhill along the slope, you eventually reach a bottom a local minimum.

### Important Terms
- **Learning Rate**: How big a step to take each update.
- **Tolerance**: How close successive steps must be to stop.
- **Local Minimum**: A point where the function value is lower than nearby points.

In this problem, Captain Redbeard finds the hidden treasure by moving downhill until he reaches the lowest point!
