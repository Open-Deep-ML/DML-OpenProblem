import math

def softmax(scores: list[float]) -> list[float]:
	sum_exp_scores = 0
	sum_exp_scores = sum(list(map(lambda a: sum_exp_scores + math.exp(a), scores)))
	probabilities = list(map(lambda a: math.exp(a)/sum_exp_scores, scores))
	return probabilities

def test_softmax():
    # Test case 1
    scores = [1, 2, 3]
    expected_output = [0.0900, 0.2447, 0.6652]
    assert softmax(scores) == expected_output, "Test case 1 failed"

    # Test case 2
    scores = [1, 1, 1]
    expected_output = [0.3333, 0.3333, 0.3333]
    assert softmax(scores) == expected_output, "Test case 2 failed"

    # Test case 3
    scores = [-1, 0, 5]
    expected_output = [0.0025, 0.0067, 0.9909]
    assert softmax(scores) == expected_output, "Test case 3 failed"

if __name__ == "__main__":
    test_softmax()
    print("All softmax tests passed.")
