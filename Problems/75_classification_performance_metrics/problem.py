def performance_metrics(actual: list[int],predicted: list[int]) -> tuple:
    #Remember to round the metrics to 3 significant digits
    return confusion_matrix, accuracy, f1Score,specificity,negativePredictiveValue