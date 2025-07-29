import math

def gaussian_probability(x, mu, sigma):
    """
    Calculate the probability of occurrence of x in a Gaussian distribution
    with the given mean (mu) and standard deviation (sigma).
    """
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
    probability = coefficient * math.exp(exponent)      

    return round(probability, 3)