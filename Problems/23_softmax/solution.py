import numpy as np

def softmax(scores):
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores)
    probabilities = np.round(exp_scores / sum_exp_scores, 4)
    return probabilities.tolist()

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
