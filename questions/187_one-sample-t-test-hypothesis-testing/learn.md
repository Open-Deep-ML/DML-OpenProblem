A one-sample t-test assesses whether a population mean differs from a hypothesized value when the population standard deviation is unknown. It uses the sample standard deviation and Student's t distribution with df = n − 1.

Test statistic:
- t = (x̄ − μ0) / (s / √n)
  - x̄: sample mean
  - μ0: hypothesized mean under H0
  - s: sample standard deviation (ddof = 1)
  - n: sample size

Tail selection from hypotheses:
- If H1 is "mu != μ0", it's two-sided: p = 2 · min(Tcdf(t), 1 − Tcdf(t))
- If H1 is "mu > μ0", it's right-tailed: p = 1 − Tcdf(t)
- If H1 is "mu < μ0", it's left-tailed: p = Tcdf(t)

Decision rule at level α:
- Reject H0 if p ≤ α; otherwise, fail to reject H0.

When to use:
- Use the t-test when σ is unknown and the sample is reasonably normal or n is moderate/large.

