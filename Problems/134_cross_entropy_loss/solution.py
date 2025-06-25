import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray) -> float:
 
    #Given
    epsilon = 1e-15
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    
    #Write your code here
    log_probs = np.log(predicted_probs)
    loss = -np.sum(true_labels * log_probs, axis=1)
    return float(np.mean(loss))

def test_compute_cross_entropy_loss():
    # Test case 1: Perfect predictions
    pred1 = np.array([[1, 0, 0], [0, 1, 0]])
    true1 = np.array([[1, 0, 0], [0, 1, 0]])
    expected1 = 0.0
    assert np.isclose(compute_cross_entropy_loss(pred1, true1), expected1), "Test case 1 failed"

    # Test case 2: Completely wrong predictions
    pred2 = np.array([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])
    true2 = np.array([[0, 0, 1], [0, 1, 0]])
    expected2 = -np.mean([np.log(0.1), np.log(0.1)])
    assert np.isclose(compute_cross_entropy_loss(pred2, true2), expected2), "Test case 2 failed"

    # Test case 3: Typical predictions
    pred3 = np.array([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]])
    true3 = np.array([[1, 0, 0], [0, 1, 0]])
    expected3 = -np.mean([np.log(0.7), np.log(0.6)])
    assert np.isclose(compute_cross_entropy_loss(pred3, true3), expected3), "Test case 3 failed"

if __name__ == "__main__":
    test_compute_cross_entropy_loss()
    print("All test cases passed!")