
from collections import Counter

def confusion_matrix(data):
    # Count all occurrences
    counts = Counter(tuple(pair) for pair in data)
    # Get metrics
    TP, FN, FP, TN = counts[(1, 1)], counts[(1, 0)], counts[(0, 1)], counts[(0, 0)]
    # Define matrix and return
    confusion_matrix = [[TP, FN], [FP, TN]]
    return confusion_matrix
