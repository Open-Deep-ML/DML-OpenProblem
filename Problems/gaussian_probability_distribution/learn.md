## Understanding Gaussian Probability Distribution

The Gaussian (or normal) distribution is a continuous probability distribution characterized by its bell-shaped curve, defined by the mean (μ) and standard deviation (σ). It describes how values are distributed around the mean, with most values clustering near the mean and fewer appearing as you move further away.

The probability density function (PDF) for a Gaussian distribution is:

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

where:
- $x$ is the value,
- $\mu$ is the mean,
- $\sigma$ is the standard deviation.

### Step-by-step Explanation

1. **Definition of the Normal Distribution**  
   The normal (Gaussian) distribution describes how values are distributed around a mean. It is symmetric and bell-shaped.

2. **Parameters**  
   - Mean (`μ`): The center of the distribution.
   - Standard deviation (`σ`): Measures the spread (width) of the distribution.

3. **General Form of the PDF**  
   The probability density function for a normal distribution is:
   $$
   f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
   $$

4. **Breakdown of the Formula**  
   - $\frac{1}{\sqrt{2\pi\sigma^2}}$: Normalizes the area under the curve to 1.
   - $\exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$: Determines the probability density at point `x`.

5. **Step-by-Step Derivation**  
   a. **Start with the standard normal distribution** (mean 0, std 1):  
      $$
      f(z) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{z^2}{2} \right)
      $$
   b. **Transform to general normal distribution**:  
      Substitute $z = \frac{x - \mu}{\sigma}$, so $x = \mu + z\sigma$.
   c. **Adjust for scaling**:  
      The PDF must be divided by $\sigma$ to account for the change in variable:
      $$
      f(x) = \frac{1}{\sigma} f\left( \frac{x - \mu}{\sigma} \right)
      $$
   d. **Plug in the standard normal PDF**:
      $$
      f(x) = \frac{1}{\sigma} \cdot \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2 \right)
      $$
   e. **Combine terms**:
      $$
      f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
      $$

6. **Interpretation**  
   This formula gives the probability density at any value `x` for a normal distribution with mean `μ` and standard deviation `σ`.

