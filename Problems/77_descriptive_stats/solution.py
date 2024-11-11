import numpy as np
from collections import Counter


def descriptive_statistics(data):
    """
    Calculate descriptive statistics: mean, median, mode, variance, standard deviation,
    percentiles, quartiles, and interquartile range for a given dataset.

    :param data: List of numerical values
    :return: Dictionary containing calculated descriptive statistics
    """
    # Convert data to numpy array for convenience
    data = np.array(data)

    # Calculate mean
    mean = np.mean(data)

    # Calculate median
    median = np.median(data)

    # Calculate mode (returns the smallest mode if multiple values)
    mode_data = Counter(data)
    mode = [k for k, v in mode_data.items() if v == max(mode_data.values())]

    # Calculate variance
    variance = np.var(data, ddof=1)  # Using ddof=1 for sample variance

    # Calculate standard deviation
    std_dev = np.std(data, ddof=1)  # Using ddof=1 for sample standard deviation

    # Calculate percentiles and quartiles
    percentiles = {p: np.percentile(data, p) for p in [25, 50, 75]}
    quartiles = {
        'Q1': percentiles[25],
        'Median': percentiles[50],
        'Q3': percentiles[75]
    }

    # Calculate interquartile range
    iqr = quartiles['Q3'] - quartiles['Q1']

    # Compile results in dictionary
    stats = {
        'Mean': round(mean, 5),
        'Median': round(median, 5),
        'Mode': mode,
        'Variance': round(variance, 5),
        'Standard Deviation': round(std_dev, 5),
        'Percentiles': percentiles,
        'Quartiles': quartiles,
        'Interquartile Range': round(iqr, 5)
    }

    return stats


def test_descriptive_statistics():
    # Test case 1: Simple dataset
    data_1 = [12, 15, 12, 18, 19, 17, 15, 14, 16, 18]
    expected_output_1 = {
        'Mean': 15.6,
        'Median': 15.5,
        'Mode': [12, 15, 18],
        'Variance': 6.93,
        'Standard Deviation': 2.63,
        'Percentiles': {25: 13.5, 50: 15.5, 75: 17.5},
        'Quartiles': {'Q1': 13.5, 'Median': 15.5, 'Q3': 17.5},
        'Interquartile Range': 4.0
    }

    output_1 = descriptive_statistics(data_1)
    assert np.isclose(output_1['Mean'], expected_output_1['Mean'], atol=1e-5), f"Mean test failed: {output_1['Mean']}"
    assert np.isclose(output_1['Median'], expected_output_1['Median'],
                      atol=1e-5), f"Median test failed: {output_1['Median']}"
    assert output_1['Mode'] == expected_output_1['Mode'], f"Mode test failed: {output_1['Mode']}"
    assert np.isclose(output_1['Variance'], expected_output_1['Variance'],
                      atol=1e-5), f"Variance test failed: {output_1['Variance']}"
    assert np.isclose(output_1['Standard Deviation'], expected_output_1['Standard Deviation'],
                      atol=1e-5), f"Standard Deviation test failed: {output_1['Standard Deviation']}"
    assert np.isclose(output_1['Interquartile Range'], expected_output_1['Interquartile Range'],
                      atol=1e-5), f"IQR test failed: {output_1['Interquartile Range']}"

    # Test case 2: Dataset with no repeated values
    data_2 = [20, 25, 30, 35, 40, 45, 50, 55, 60]
    expected_output_2 = {
        'Mean': 40.0,
        'Median': 40.0,
        'Mode': [],  # No mode for unique values
        'Variance': 166.66667,
        'Standard Deviation': 12.90994,
        'Percentiles': {25: 32.5, 50: 40.0, 75: 47.5},
        'Quartiles': {'Q1': 32.5, 'Median': 40.0, 'Q3': 47.5},
        'Interquartile Range': 15.0
    }

    output_2 = descriptive_statistics(data_2)
    assert np.isclose(output_2['Mean'], expected_output_2['Mean'], atol=1e-5), f"Mean test failed: {output_2['Mean']}"
    assert np.isclose(output_2['Median'], expected_output_2['Median'],
                      atol=1e-5), f"Median test failed: {output_2['Median']}"
    assert output_2['Mode'] == expected_output_2['Mode'], f"Mode test failed: {output_2['Mode']}"
    assert np.isclose(output_2['Variance'], expected_output_2['Variance'],
                      atol=1e-5), f"Variance test failed: {output_2['Variance']}"
    assert np.isclose(output_2['Standard Deviation'], expected_output_2['Standard Deviation'],
                      atol=1e-5), f"Standard Deviation test failed: {output_2['Standard Deviation']}"
    assert np.isclose(output_2['Interquartile Range'], expected_output_2['Interquartile Range'],
                      atol=1e-5), f"IQR test failed: {output_2['Interquartile Range']}"

    print("All descriptive statistics tests passed.")


if __name__ == "__main__":
    test_descriptive_statistics()
