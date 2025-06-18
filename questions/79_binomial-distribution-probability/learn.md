## Understanding the Binomial Distribution

The Binomial distribution is a discrete probability distribution that models the number of successes in a fixed number of independent Bernoulli trials, each with the same probability of success.

### Mathematical Formulation

The probability of achieving exactly $k$ successes in $n$ trials is given by the formula:

$$
P(X = k) = \binom{n}{k} \cdot p^k \cdot (1-p)^{n-k}
$$

- **$n$**: Total number of trials  
- **$k$**: Number of successes  
- **$p$**: Probability of success on each trial  
- $\binom{n}{k}$: The number of ways to choose $k$ successes from $n$ trials, calculated as:

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}
$$

### Implementation Steps

1. Calculate $\binom{n}{k}$ using factorials.  
2. Raise $p$ to the power of $k$ and $(1-p)$ to the power of $(n-k)$.  
3. Multiply these results to get the probability.

### Example Calculation

Given:

- $n = 5$  
- $k = 2$  
- $p = 0.4$  

Step-by-step:

1. Calculate $\binom{n}{k}$:

$$
\binom{5}{2} = \frac{5!}{2!(5-2)!} = \frac{5 \cdot 4}{2 \cdot 1} = 10
$$

2. Calculate $p^k \cdot (1-p)^{n-k}$:

$$
0.4^2 \cdot (1-0.4)^3 = 0.16 \cdot 0.216 = 0.03456
$$

3. Multiply results:

$$
P(X = 2) = 10 \cdot 0.03456 = 0.3456
$$

The probability of exactly 2 successes is $0.3456$.

### Applications

The Binomial distribution is widely used in:

- Quality control and defect analysis  
- Survey analysis  
- Medical trials  
- Modeling success/failure experiments  

It provides insights into the likelihood of various outcomes in scenarios with two possible results (e.g., success or failure).
