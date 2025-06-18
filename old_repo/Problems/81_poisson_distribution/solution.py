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

def test_poisson_probability():
    # Test case 1: Basic Poisson distribution calculation
    k1, lam1 = 3, 5
    expected_output_1 = 0.14037
    output_1 = poisson_probability(k1, lam1)
    assert output_1 == expected_output_1, \
        f"Test case 1 failed: expected {expected_output_1}, got {output_1}"

    # Test case 2: Calculate probability for k = 0
    k2, lam2 = 0, 5
    expected_output_2 = 0.00674
    output_2 = poisson_probability(k2, lam2)
    assert output_2 == expected_output_2, \
        f"Test case 2 failed: expected {expected_output_2}, got {output_2}"

    # Test case 3: Larger lambda (mean rate)
    k3, lam3 = 2, 10
    expected_output_3 = 0.00045
    output_3 = poisson_probability(k3, lam3)
    assert output_3 == expected_output_3, \
        f"Test case 3 failed: expected {expected_output_3}, got {output_3}"

    # Test case 4: Small k with small lambda
    k4, lam4 = 1, 1
    expected_output_4 = 0.36788
    output_4 = poisson_probability(k4, lam4)
    assert output_4 == expected_output_4, \
        f"Test case 4 failed: expected {expected_output_4}, got {output_4}"

    # Test case 5: Larger k with large lambda
    k5, lam5 = 20, 20
    expected_output_5 = 0.08505
    output_5 = poisson_probability(k5, lam5)
    assert output_5 == expected_output_5, \
        f"Test case 5 failed: expected {expected_output_5}, got {output_5}"

    print("All Poisson distribution tests passed.")

if __name__ == "__main__":
    test_poisson_probability()
