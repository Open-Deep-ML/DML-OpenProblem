
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	# Implement your code here
	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
