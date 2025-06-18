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
    probability = binomial_coeff * (p ** k) * ((1 - p) ** (n - k))
    # Return the probability, rounded to five decimal places
    return round(probability, 5)

def test_binomial_probability():
    # Test case 1: Basic binomial calculation
    n1, k1, p1 = 6, 2, 0.5
    expected_output_1 = 0.23438
    output_1 = binomial_probability(n1, k1, p1)
    assert output_1 == expected_output_1, \
        f"Test case 1 failed: expected {expected_output_1}, got {output_1}"

    # Test case 2: Higher success probability
    n2, k2, p2 = 6, 4, 0.7
    expected_output_2 = 0.32413
    output_2 = binomial_probability(n2, k2, p2)
    assert output_2 == expected_output_2, \
        f"Test case 2 failed: expected {expected_output_2}, got {output_2}"

    # Test case 3: All trials succeed
    n3, k3, p3 = 3, 3, 0.9
    expected_output_3 = 0.729
    output_3 = binomial_probability(n3, k3, p3)
    assert output_3 == expected_output_3, \
        f"Test case 3 failed: expected {expected_output_3}, got {output_3}"

    # Test case 4: All trials fail
    n4, k4, p4 = 5, 0, 0.3
    expected_output_4 = 0.16807
    output_4 = binomial_probability(n4, k4, p4)
    assert output_4 == expected_output_4, \
        f"Test case 4 failed: expected {expected_output_4}, got {output_4}"

    # Test case 5: Low probability of success
    n5, k5, p5 = 7, 2, 0.1
    expected_output_5 = 0.12106
    output_5 = binomial_probability(n5, k5, p5)
    assert output_5 == expected_output_5, \
        f"Test case 5 failed: expected {expected_output_5}, got {output_5}"

    print("All Binomial distribution tests passed.")

if __name__ == "__main__":
    test_binomial_probability()
