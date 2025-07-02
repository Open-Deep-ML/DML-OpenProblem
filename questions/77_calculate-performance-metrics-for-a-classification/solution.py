from collections import Counter


def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    data = list(zip(actual, predicted))
    counts = Counter(tuple(pair) for pair in data)
    TP, FN, FP, TN = counts[(1, 1)], counts[(1, 0)], counts[(0, 1)], counts[(0, 0)]
    confusion_matrix = [[TP, FN], [FP, TN]]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    negativePredictive = TN / (TN + FN)
    specificity = TN / (TN + FP)
    return (
        confusion_matrix,
        round(accuracy, 3),
        round(f1, 3),
        round(specificity, 3),
        round(negativePredictive, 3),
    )
