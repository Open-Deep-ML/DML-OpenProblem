import math

def poisson_probability(k, lam):
    """
    Calculate the probability of observing exactly k events in a fixed interval,
    given the mean rate of events lam, using the Poisson distribution formula.
    :param k: Number of events (non-negative integer)
    :param lam: The average rate (mean) of occurrences in a fixed interval
    :return: Probability of k events occurring
    """
    # Calculate the Poisson probability using the formula
    probability = (lam ** k) * math.exp(-lam) / math.factorial(k)
    # Return the probability, rounded to five decimal places
    return round(probability, 5)
