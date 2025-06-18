import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    weights = np.array(initial_weights)
    bias = initial_bias
    features = np.array(features)
    labels = np.array(labels)
    mse_values = []

    for _ in range(epochs):
        z = np.dot(features, weights) + bias
        predictions = sigmoid(z)
        
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(mse, 4))

        # Gradient calculation for weights and bias
        errors = predictions - labels
        weight_gradients = (2/len(labels)) * np.dot(features.T, errors * predictions * (1 - predictions))
        bias_gradient = (2/len(labels)) * np.mean(errors * predictions * (1 - predictions))
        
        # Update weights and bias
        weights -= learning_rate * weight_gradients
        bias -= learning_rate * bias_gradient

        # Round weights and bias for output
        updated_weights = np.round(weights, 4)
        updated_bias = round(bias, 4)

    return updated_weights.tolist(), updated_bias, mse_values

def test_train_neuron():
    # Test case 1
    features = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]])
    labels = np.array([1, 0, 0])
    initial_weights = np.array([0.1, -0.2])
    initial_bias = 0.0
    learning_rate = 0.1
    epochs = 2
    expected_output = ([0.1035, -0.1426], -0.0056, [0.3033, 0.2947])
    assert train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs) == expected_output, "Test case 1 failed"
    
    # Test case 2
    features = np.array([[1, 2], [2, 3], [3, 1]])
    labels = np.array([1, 0, 1])
    initial_weights = np.array([0.5, -0.2])
    initial_bias = 0.0
    learning_rate = 0.1
    epochs = 3
    expected_output = ([0.4893, -0.2301], 0.001, [0.21, 0.2087, 0.2076])
    assert train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs) == expected_output, "Test case 2 failed"

if __name__ == "__main__":
    test_train_neuron()
    print("All train_neuron tests passed.")
