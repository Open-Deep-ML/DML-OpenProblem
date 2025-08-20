## Understanding Chi-Square Probability Distribution

The Chi-square distribution is a continuous probability distribution that arises in statistics when estimating the variance of a normally distributed population. It is defined by a single parameter: the degrees of freedom ($k$), which is typically a positive integer.

The probability density function (PDF) for a Chi-square distribution is:

$$
f(x; k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{(k/2) - 1} e^{-x/2}
$$

where:
- $x \geq 0$ is the value,
- $k$ is the degrees of freedom,
- $\Gamma$ is the gamma function.

### Step-by-step Explanation

1. **Definition of the Chi-square Distribution**  
        The Chi-square distribution describes the distribution of a sum of the squares of $k$ independent standard normal random variables.

2. **Parameter**  
        - Degrees of freedom ($k$): Determines the shape of the distribution.

3. **General Form of the PDF**  
        The probability density function for a Chi-square distribution is:
        $$
        f(x; k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{(k/2) - 1} e^{-x/2}
        $$

4. **Breakdown of the Formula**  
        - $\frac{1}{2^{k/2} \Gamma(k/2)}$: Normalizes the area under the curve to 1.
        - $x^{(k/2) - 1}$: Adjusts the shape based on degrees of freedom.
        - $e^{-x/2}$: Ensures the distribution decays for large $x$.

5. **Step-by-Step Derivation**  
    a) **Start with standard normal variables:**  
        Let $Z_1, Z_2, \ldots, Z_k$ be independent standard normal random variables.

    b) **Sum of squares:**  
        Define $X = Z_1^2 + Z_2^2 + \ldots + Z_k^2$. The variable $X$ then follows a Chi-square distribution with $k$ degrees of freedom.

    c) **PDF derivation:**  
        By applying transformation techniques and properties of the gamma function, the probability density function for $X$ is derived as:
        $$
        f(x; k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{(k/2) - 1} e^{-x/2}
        $$
        This formula results from the transformation and the gamma function properties.

6. **Interpretation**  
        This formula gives the probability density at any value $x$ for a Chi-square distribution with $k$ degrees of freedom.

### Uses and Applications

- **Hypothesis Testing**: Used in the Chi-square test for independence and goodness-of-fit tests.
- **Confidence Intervals**: Helps construct confidence intervals for population variance.
- **Model Fitting**: Assesses how well observed data fit a statistical model.
- **Variance Analysis**: Common in analysis of variance (ANOVA) and regression diagnostics.

