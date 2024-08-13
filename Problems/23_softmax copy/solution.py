import math

def single_neuron_model(features, labels, weights, bias):
    probabilities = []
    for feature_vector in features:
        z = sum(weight * feature for weight, feature in zip(weights, feature_vector)) + bias
        prob = 1 / (1 + math.exp(-z))
        probabilities.append(round(prob, 4))
    
    mse = sum((prob - label) ** 2 for prob, label in zip(probabilities, labels)) / len(labels)
    mse = round(mse, 4)
    
    return probabilities, mse

def test_single_neuron_model():
    # Test case 1
    features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
    labels = [0, 1, 0]
    weights = [0.7, -0.4]
    bias = -0.1
    expected_output = ([0.4626, 0.4134, 0.6682], 0.3349)
    assert single_neuron_model(features, labels, weights, bias) == expected_output, "Test case 1 failed"
    
    # Test case 2
    features = [[1, 2], [2, 3], [3, 1]]
    labels = [1, 0, 1]
    weights = [0.5, -0.2]
    bias = 0
    expected_output = ([0.525, 0.5987, 0.7858], 0.21)
    assert single_neuron_model(features, labels, weights, bias) == expected_output, "Test case 2 failed"

if __name__ == "__main__":
    test_single_neuron_model()
    print("All single_neuron_model tests passed.")
