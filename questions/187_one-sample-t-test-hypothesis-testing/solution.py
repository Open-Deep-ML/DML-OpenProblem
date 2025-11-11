from math import gamma, sqrt, pi

def _t_pdf(x, df):
    c = gamma((df + 1) / 2.0) / (sqrt(df * pi) * gamma(df / 2.0))
    return c * (1.0 + (x * x) / df) ** (-(df + 1) / 2.0)

def _t_cdf(x, df):
    if x == 0.0:
        return 0.5
    sign = 1.0 if x > 0 else -1.0
    a, b = 0.0, abs(x)
    n = 4000  # even number of intervals; higher for better accuracy
    h = (b - a) / n
    s = _t_pdf(a, df) + _t_pdf(b, df)
    for i in range(1, n):
        xi = a + i * h
        s += (4 if i % 2 == 1 else 2) * _t_pdf(xi, df)
    integral = s * h / 3.0
    return 0.5 + sign * integral

def _parse_hypotheses(H0, H1):
    text = H0.replace(" ", "")
    if "=" not in text:
        raise ValueError("H0 must specify equality, e.g., 'mu = 100'")
    mu0 = float(text.split("=")[1])
    alt_text = H1.replace(" ", "").replace("<>", "!=").replace("≠", "!=")
    if "!=" in alt_text:
        alt = "two-sided"
    elif ">" in alt_text:
        alt = "greater"
    elif "<" in alt_text:
        alt = "less"
    else:
        alt = "two-sided"
    return mu0, alt

def one_sample_t_test(sample_mean, sample_std, n, H0, H1):
    """
    Perform a one-sample t-test for a population mean with unknown population std.
    Auto-detect tail from H1 and extract μ0 from H0.
    Returns dict with keys: "t", "df", "p_value", "alternative".
    """
    mu0, alternative = _parse_hypotheses(H0, H1)
    df = n - 1
    se = sample_std / sqrt(n)
    t = (sample_mean - mu0) / se
    cdf = _t_cdf(t, df)

    if alternative == "two-sided":
        p = 2.0 * min(cdf, 1.0 - cdf)
    elif alternative == "greater":
        p = 1.0 - cdf
    elif alternative == "less":
        p = cdf
    else:
        p = 2.0 * min(cdf, 1.0 - cdf)

    return {"t": round(t, 4), "df": df, "p_value": round(p, 4), "alternative": alternative}

