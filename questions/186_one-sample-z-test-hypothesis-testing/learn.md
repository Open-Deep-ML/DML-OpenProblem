A one-sample Z-test assesses whether the mean of a population differs from a hypothesized value when the population standard deviation is known. It is appropriate for large samples (by CLT) or when normality is assumed and the population standard deviation is known.

Test statistic:
- z = (x̄ − μ0) / (σ / √n)
  - x̄: sample mean
  - μ0: hypothesized mean under H0
  - σ: known population standard deviation
  - n: sample size

P-value computation uses the standard normal distribution:
- Two-sided (H1: μ ≠ μ0): p = 2 · min(Φ(z), 1 − Φ(z))
- Right-tailed (H1: μ > μ0): p = 1 − Φ(z)
- Left-tailed (H1: μ < μ0): p = Φ(z)

Decision at level α:
- Reject H0 if p ≤ α; otherwise, fail to reject H0.

Notes:
- If σ is unknown, use a one-sample t-test with the sample standard deviation instead.

