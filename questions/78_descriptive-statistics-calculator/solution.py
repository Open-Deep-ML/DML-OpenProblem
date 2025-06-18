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
        "variance": np.round(variance,4),
        "standard_deviation": np.round(std_dev,4),
        "25th_percentile": percentiles[0],
        "50th_percentile": percentiles[1],
        "75th_percentile": percentiles[2],
        "interquartile_range": iqr
    }

    return stats_dict
