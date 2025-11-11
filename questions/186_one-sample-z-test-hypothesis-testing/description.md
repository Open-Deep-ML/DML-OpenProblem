Implement a function to perform a one-sample Z-test for a population mean when the population standard deviation is known. Your function must support both one-tailed and two-tailed alternatives.

Implement a function with the signature:
- one_sample_z_test(sample_mean, population_mean, population_std, n, alternative="two-sided")

Where:
- sample_mean: The observed sample mean (float)
- population_mean: The hypothesized population mean under H0 (float)
- population_std: The known population standard deviation (float > 0)
- n: Sample size (int > 0)
- alternative: One of {"two-sided", "greater", "less"}

Return a dictionary with:
- "z": the computed Z statistic rounded to 4 decimals
- "p_value": the corresponding p-value rounded to 4 decimals

Use the standard normal distribution for the p-value. Handle invalid inputs minimally by assuming valid types and values.

