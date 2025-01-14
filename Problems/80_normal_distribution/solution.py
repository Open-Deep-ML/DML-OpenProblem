import math

def normal_pdf(x, mean, std_dev):
    """
    Calculate the probability density function (PDF) of the normal distribution.

    :param x: The value at which the PDF is evaluated.
    :param mean: The mean (μ) of the distribution.
    :param std_dev: The standard deviation (σ) of the distribution.
    :return: The PDF value for the given x.
    """
    coefficient = 1 / (math.sqrt(2 * math.pi) * std_dev)
    exponent = math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
    return round(coefficient * exponent, 5)

def test_normal_pdf():
    """
    Test cases for the normal_pdf function.
    """
    # Test case 1: Standard normal distribution
    x1, mean1, std_dev1 = 0, 0, 1
    expected_output_1 = 0.39894
    output_1 = normal_pdf(x1, mean1, std_dev1)
    assert output_1 == expected_output_1, \
        f"Test case 1 failed: expected {expected_output_1}, got {output_1}"

    # Test case 2: Normal distribution with non-zero mean
    x2, mean2, std_dev2 = 16, 15, 2.04
    expected_output_2 = 0.17603
    output_2 = normal_pdf(x2, mean2, std_dev2)
    assert output_2 == expected_output_2, \
        f"Test case 2 failed: expected {expected_output_2}, got {output_2}"

    # Test case 3: Low standard deviation
    x3, mean3, std_dev3 = 1, 0, 0.5
    expected_output_3 = 0.10798
    output_3 = normal_pdf(x3, mean3, std_dev3)
    assert output_3 == expected_output_3, \
        f"Test case 3 failed: expected {expected_output_3}, got {output_3}"

    print("All Normal Distribution tests passed.")

if __name__ == "__main__":
    # Run the test cases
    test_normal_pdf()