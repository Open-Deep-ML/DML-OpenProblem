def min_max(x: list[int]) -> list[float]:
    largest = max(x)
    smallest = min(x)
    if largest == smallest:
        return [0.0] * len(x)
    for i in range(len(x)):
        x[i] = round((x[i] - smallest) / (largest - smallest), 4)
    return x
