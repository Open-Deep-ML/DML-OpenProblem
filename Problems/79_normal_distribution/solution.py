import math


def normal_distribution_probability(mean, std_dev, x):
    """
    Calculate the probability density of a given x-value in a normal distribution
    with specified mean and standard deviation.

    :param mean: The mean of the distribution
    :param std_dev: The standard deviation of the distribution
    :param x: The value for which we want the probability density
    :return: Probability density at x
    """
    # Use the PDF formula for the normal distribution
    probability_density = (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
    return round(probability_density, 5)


# Test cases
def test_normal_distribution():
    # Test case 1: Simple test for probability density at the mean
    mean, std_dev, x = 0, 1, 0
    expected_density = 0.39894
    output_density = normal_distribution_probability(mean, std_dev, x)
    assert abs(
        output_density - expected_density) < 1e-5, f"Density test failed: expected {expected_density}, got {output_density}"

    # Test case 2: Probability density for a value one standard deviation from the mean
    mean, std_dev, x = 0, 1, 1
    expected_density = 0.24197
    output_density = normal_distribution_probability(mean, std_dev, x)
    assert abs(
        output_density - expected_density) < 1e-5, f"Density test failed: expected {expected_density}, got {output_density}"

    # Test case 3: Probability density for a value two standard deviations from the mean
    mean, std_dev, x = 0, 1, 2
    expected_density = 0.05399
    output_density = normal_distribution_probability(mean, std_dev, x)
    assert abs(
        output_density - expected_density) < 1e-5, f"Density test failed: expected {expected_density}, got {output_density}"

    # Test case 4: Probability density for a negative x value
    mean, std_dev, x = 0, 1, -1
    expected_density = 0.24197
    output_density = normal_distribution_probability(mean, std_dev, x)
    assert abs(
        output_density - expected_density) < 1e-5, f"Density test failed: expected {expected_density}, got {output_density}"

    print("All normal distribution tests passed.")


if __name__ == "__main__":
    test_normal_distribution()
