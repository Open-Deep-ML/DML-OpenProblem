import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
    m = len(y)
    
    for _ in range(n_iterations):
        if method == 'batch':
            # Calculate the gradient using all data points
            predictions = X.dot(weights)
            errors = predictions - y
            gradient = 2 * X.T.dot(errors) / m
            weights = weights - learning_rate * gradient
        
        elif method == 'stochastic':
            # Update weights for each data point individually
            for i in range(m):
                prediction = X[i].dot(weights)
                error = prediction - y[i]
                gradient = 2 * X[i].T.dot(error)
                weights = weights - learning_rate * gradient
        
        elif method == 'mini_batch':
            # Update weights using sequential batches of data points without shuffling
            for i in range(0, m, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                predictions = X_batch.dot(weights)
                errors = predictions - y_batch
                gradient = 2 * X_batch.T.dot(errors) / batch_size
                weights = weights - learning_rate * gradient
                
    return weights

def test_gradient_descent():
    # Test case 1: Batch Gradient Descent
    X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
    y = np.array([2, 3, 4, 5])
    weights = np.zeros(X.shape[1])
    learning_rate = 0.01
    n_iterations = 100
    expected_output = np.array([1.14905239, 0.56176776])
    assert np.allclose(gradient_descent(X, y, weights, learning_rate, n_iterations, method='batch'), expected_output), "Test case 1 failed"
    
    # Test case 2: Stochastic Gradient Descent
    weights = np.zeros(X.shape[1])
    expected_output = np.array([1.0507814, 0.83659454])
    assert np.allclose(gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic'), expected_output), "Test case 2 failed"
    
    # Test case 3: Mini-Batch Gradient Descent
    weights = np.zeros(X.shape[1])
    batch_size = 2
    expected_output = np.array([1.10334065, 0.68329431])
    assert np.allclose(gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size, method='mini_batch'), expected_output), "Test case 3 failed"

if __name__ == "__main__":
    test_gradient_descent()
    print("All gradient_descent tests passed.")
