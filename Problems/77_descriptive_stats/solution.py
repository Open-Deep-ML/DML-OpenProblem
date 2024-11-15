import numpy as np


def descriptive_statistics(data):
    """
    Calculate various descriptive statistics metrics for a given dataset.

    :param data: List or numpy array of numerical values
    :return: Dictionary containing mean, median, mode, variance, standard deviation,
             percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Ensure data is a numpy array for easier calculations
    data = np.array(data)

    # Mean
    mean = np.mean(data)

    # Median
    median = np.median(data)

    # Mode
    unique, counts = np.unique(data, return_counts=True)
    mode = unique[np.argmax(counts)] if len(data) > 0 else None

    # Variance
    variance = np.var(data)

    # Standard Deviation
    std_dev = np.sqrt(variance)

    # Percentiles (25th, 50th, 75th)
    percentiles = np.percentile(data, [25, 50, 75])

    # Interquartile Range (IQR)
    iqr = percentiles[2] - percentiles[0]

    # Compile results into a dictionary
    stats_dict = {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": variance,
        "standard_deviation": std_dev,
        "25th_percentile": percentiles[0],
        "50th_percentile": percentiles[1],
        "75th_percentile": percentiles[2],
        "interquartile_range": iqr
    }

    return stats_dict


def test_descriptive_statistics():
    # Test case 1: Simple dataset
    data_1 = [10, 20, 30, 40, 50]
    expected_output_1 = {
        "mean": 30.0,
        "median": 30.0,
        "mode": 10,  # assuming the smallest element if no mode
        "variance": 200.0,
        "standard_deviation": 14.142135623730951,
        "25th_percentile": 20.0,
        "50th_percentile": 30.0,
        "75th_percentile": 40.0,
        "interquartile_range": 20.0
    }
    output_1 = descriptive_statistics(data_1)
    assert all(np.isclose(output_1[key], value, atol=1e-5) for key, value in expected_output_1.items()), \
        f"Test case 1 failed: expected {expected_output_1}, got {output_1}"

    # Test case 2: Dataset with repeated elements
    data_2 = [1, 2, 2, 3, 4, 4, 4, 5]
    expected_output_2 = {
        "mean": 3.125,
        "median": 3.5,
        "mode": 4,
        "variance": 1.609375,
        "standard_deviation": 1.268857754044952,
        "25th_percentile": 2.0,
        "50th_percentile": 3.5,
        "75th_percentile": 4.0,
        "interquartile_range": 2.0
    }
    output_2 = descriptive_statistics(data_2)
    assert all(np.isclose(output_2[key], value, atol=1e-5) for key, value in expected_output_2.items()), \
        f"Test case 2 failed: expected {expected_output_2}, got {output_2}"

    # Test case 3: Single-element dataset
    data_3 = [100]
    expected_output_3 = {
        "mean": 100.0,
        "median": 100.0,
        "mode": 100,
        "variance": 0.0,
        "standard_deviation": 0.0,
        "25th_percentile": 100.0,
        "50th_percentile": 100.0,
        "75th_percentile": 100.0,
        "interquartile_range": 0.0
    }
    output_3 = descriptive_statistics(data_3)
    assert all(np.isclose(output_3[key], value, atol=1e-5) for key, value in expected_output_3.items()), \
        f"Test case 3 failed: expected {expected_output_3}, got {output_3}"

    print("All descriptive statistics tests passed.")


if __name__ == "__main__":
    test_descriptive_statistics()
