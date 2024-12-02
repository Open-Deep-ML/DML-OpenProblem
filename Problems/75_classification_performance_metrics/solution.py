from collections import Counter

def performance_metrics(actual: list[int],predicted: list[int]) -> tuple:
    #Merge lists into one
    data = list(zip(actual,predicted))
    #Count all occurrences
    counts = Counter(tuple(pair) for pair in data)
    #Get metrics
    TP, FN, FP, TN = counts[(1,1)],counts[(1,0)],counts[(0,1)],counts[(0,0)]
    #Define confusin matrix
    confusion_matrix = [[TP,FN],[FP,TN]]
    #Get accuracy
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    #Get precision
    precision = TP/(TP+FP)
    #Get recall
    recall = TP/(TP + FN)
    #Get F1 score
    f1 = 2*precision*recall/(precision+recall)
    #Get negative predictive value
    negativePredictive = TN/(TN+FN)
    #Get specificiy
    specificity = TN/(TN+FP)
    #Return
    return confusion_matrix, round(accuracy,3),round(f1,3), round(specificity,3), round(negativePredictive,3), 

def test_confusion_matrix() -> None:
    # Test case 1
    y_actual, y_pred = [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0]
    assert performance_metrics(y_actual,y_pred) == ([[6, 4], [2, 7]],0.684,  0.667, 0.778, 0.636), "Test case 1 failed"

    # Test case 2
    y_actual, y_pred = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],[1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    assert performance_metrics(y_actual,y_pred) == ([[4, 4], [5, 2]], 0.4, 0.471, 0.286, 0.333), "Test case 1 failed"

    # Test case 3
    y_actual, y_pred = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],[0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0]
    assert performance_metrics(y_actual,y_pred) == ([[4, 5], [4, 2]], 0.4, 0.471, 0.333, 0.286), "Test case 1 failed"

if __name__ == "__main__":
    test_confusion_matrix()
    print("All performance metrics tests passed.")