def phi_corr(x: list[int], y: list[int]) -> float:
    """
    Calculate the Phi coefficient between two binary variables.

    Args:
    x (list[int]): A list of binary values (0 or 1).
    y (list[int]): A list of binary values (0 or 1).

    Returns:
    float: The Phi coefficient rounded to 4 decimal places.
    """
    x1y1 = x1y0 = x0y1 = x0y0 = 0

    # Count occurrences
    for i in range(len(x)):
        if x[i] == 1:
            if y[i] == 1:
                x1y1 += 1
            else:
                x1y0 += 1
        elif x[i] == 0:
            if y[i] == 1:
                x0y1 += 1
            else:
                x0y0 += 1

    # Calculate numerator and denominator
    numerator = (x0y0 * x1y1) - (x0y1 * x1y0)
    denominator = ((x0y0 + x0y1) * (x1y0 + x1y1) * (x0y0 + x1y0) * (x0y1 + x1y1)) ** 0.5

    if denominator == 0:
        return 0.0

    phi = numerator / denominator
    return round(phi, 4)
