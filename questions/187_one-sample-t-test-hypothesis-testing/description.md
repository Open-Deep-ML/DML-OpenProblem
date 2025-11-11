Implement a function to perform a one-sample t-test for a population mean when the population standard deviation is unknown (use sample standard deviation). Your function must auto-detect whether the test is one-tailed or two-tailed by parsing the hypotheses.

Implement a function with the signature:
- one_sample_t_test(sample_mean, sample_std, n, H0, H1)

Where:
- sample_mean: Observed sample mean (float)
- sample_std: Sample standard deviation (float > 0, computed with ddof=1)
- n: Sample size (int > 1)
- H0: Null hypothesis as a string, e.g., "mu = 100"
- H1: Alternative hypothesis as a string, e.g., "mu > 100", "mu < 100", or "mu != 100"

Requirements:
- Extract the hypothesized mean μ0 from H0.
- Determine the tail from H1:
  - "mu != μ0" → two-sided
  - "mu > μ0" → right-tailed
  - "mu < μ0" → left-tailed
- Compute the t-statistic: t = (x̄ − μ0) / (s / √n)
- Degrees of freedom: df = n − 1
- Compute the p-value using the Student's t distribution.

Return a dictionary with:
- "t": computed t-statistic rounded to 4 decimals
- "df": degrees of freedom (int)
- "p_value": p-value rounded to 4 decimals
- "alternative": one of {"two-sided", "greater", "less"}

