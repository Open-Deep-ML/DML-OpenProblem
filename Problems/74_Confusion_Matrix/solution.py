from collections import Counter

def confusion_matrix(data: list[list[int|int]]) -> list[list[int|int]]:
    #Count all occurrences
    counts = Counter(tuple(pair) for pair in data)
    #Get metrics
    TP, FN, FP, TN = counts[(1,1)],counts[(1,0)],counts[(0,1)],counts[(0,0)]
    #Define matrix and return
    confusion_matrix = [[TP,FN],[FP,TN]]
    return confusion_matrix

def test_confusion_matrix() -> None:
    # Test case 1
    data = [[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]
    assert confusion_matrix(data) == [[1, 1], [2, 1]], "Test case 1 failed"

    # Test case 2
    data = [[0, 1], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [1, 1], [1, 0]]
    assert confusion_matrix(data) == [[5, 5], [4, 3]], "Test case 2 failed"

    # Test case 3
    data = [[0, 1], [0, 1], [0, 0], [0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [1, 0], [0, 1], [1, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 0]]
    assert confusion_matrix(data) == [[0, 3], [9, 4]], "Test case 3 failed"

if __name__ == "__main__":
    test_confusion_matrix()
    print("All confusion matrix tests passed.")
