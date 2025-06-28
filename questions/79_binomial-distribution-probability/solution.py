import math


def binomial_probability(n, k, p):
    """
    Calculate the probability of achieving exactly k successes in n independent Bernoulli trials,
    each with probability p of success, using the Binomial distribution formula.
    :param n: Total number of trials
    :param k: Number of successes
    :param p: Probability of success on each trial
    :return: Probability of k successes in n trials
    """
    # Calculate binomial coefficient (n choose k)
    binomial_coeff = math.comb(n, k)
    # Calculate the probability using the binomial formula
    probability = binomial_coeff * (p**k) * ((1 - p) ** (n - k))
    # Return the probability, rounded to five decimal places
    return round(probability, 5)
