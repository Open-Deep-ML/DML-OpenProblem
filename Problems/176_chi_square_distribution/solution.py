import math

def chi_square_probability(x, k):
    """
    Calculate the probability density of x in a Chi-square distribution
    with k degrees of freedom.
    """
    if x < 0 or k <= 0:
        return 0.0
    
    coeff = 1 / (math.pow(2, k / 2) * math.gamma(k / 2))
    probability = coeff * math.pow(x, (k / 2) - 1) * math.exp(-x / 2)

    return round(probability, 3)