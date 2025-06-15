## Understanding Normal Distribution

The Normal Distribution, also known as the Gaussian Distribution, is a continuous probability distribution that is symmetrical and bell-shaped, representing the distribution of data around the mean.

### Key Characteristics

- **Symmetry**: The distribution is symmetric around the mean, which means the left and right halves of the graph are mirror images.
- **Mean, Median, and Mode**: In a perfectly normal distribution, the mean, median, and mode are all equal.
- **Shape**: The bell-shaped curve is defined by its mean ($\mu$) and standard deviation ($\sigma$).
- **Empirical Rule**: Approximately:
  - $68\%$ of data falls within 1 standard deviation ($\mu \pm \sigma$).
  - $95\%$ of data falls within 2 standard deviations ($\mu \pm 2\sigma$).
  - $99.7\%$ of data falls within 3 standard deviations ($\mu \pm 3\sigma$).

### Mathematical Formula

The probability density function (PDF) of a normal distribution is given by:

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

- **$x$**: Random variable
- **$\mu$**: Mean of the distribution
- **$\sigma$**: Standard deviation of the distribution

### Implementation Steps

1. **Calculate the mean ($\mu$) and standard deviation ($\sigma$):**
   - $\mu = \frac{\sum x_i}{N}$
   - $\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{N}}$

2. **Use the normal distribution formula to calculate the probability density for each value:**
   - $\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

3. **Visualize the curve**:
   - Plot the calculated PDF values for a range of $x$ to visualize the bell-shaped curve.

### Example Calculation

Given:

- Data: [10, 12, 14, 16, 18, 20]

1. **Mean ($\mu$):**
   $$
   \mu = \frac{10 + 12 + 14 + 16 + 18 + 20}{6} = 15
   $$

2. **Standard Deviation ($\sigma$):**
   $$
   \sigma = \sqrt{\frac{(10-15)^2 + \dots + (20-15)^2}{6}} = \sqrt{\frac{25}{6}} \approx 2.04
   $$

3. **PDF for $x = 16$:**
   $$
   f(16) = \frac{1}{\sqrt{2\pi(2.04)^2}} e^{-\frac{(16-15)^2}{2(2.04)^2}} \approx 0.176
   $$

### Applications

The Normal Distribution is widely used in:

- Data Analysis
- Statistical Inference
- Machine Learning Algorithms
- Quality Control
- Risk Management

This distribution is crucial in fields like economics, biology, psychology, and engineering to model natural phenomena and make predictions.
