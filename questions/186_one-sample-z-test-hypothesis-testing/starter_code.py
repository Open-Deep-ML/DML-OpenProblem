from math import erf, sqrt

def _standard_normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def one_sample_z_test(sample_mean, population_mean, population_std, n, alternative="two-sided"):
    """
    Perform a one-sample Z-test for a population mean with known population std.

    Parameters
    ----------
    sample_mean : float
    population_mean : float
    population_std : float
    n : int
    alternative : str
        One of {"two-sided", "greater", "less"}

    Returns
    -------
    dict with keys:
      - "z": Z-statistic rounded to 4 decimals
      - "p_value": p-value rounded to 4 decimals
    """
    # TODO: Implement the Z statistic and p-value computation
    # z = (sample_mean - population_mean) / (population_std / sqrt(n))
    # Use _standard_normal_cdf for CDF of standard normal.
    # For alternative:
    # - "two-sided": p = 2 * min(P(Z<=z), P(Z>=z)) = 2 * min(cdf(z), 1-cdf(z))
    # - "greater":   p = 1 - cdf(z)
    # - "less":      p = cdf(z)
    return {"z": 0.0, "p_value": 1.0}

