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
    standard_error = population_std / sqrt(n)
    z = (sample_mean - population_mean) / standard_error
    cdf = _standard_normal_cdf(z)

    if alternative == "two-sided":
        p = 2.0 * min(cdf, 1.0 - cdf)
    elif alternative == "greater":
        p = 1.0 - cdf
    elif alternative == "less":
        p = cdf
    else:
        # Fallback to two-sided if unexpected input
        p = 2.0 * min(cdf, 1.0 - cdf)

    return {"z": round(z, 4), "p_value": round(p, 4)}

