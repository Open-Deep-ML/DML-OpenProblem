def gradient_descent(
    X, y, weights, learning_rate, n_iterations, batch_size=1, method="batch"
):
    m = len(y)

    for _ in range(n_iterations):
        if method == "batch":
            # Calculate the gradient using all data points
            predictions = X.dot(weights)
            errors = predictions - y
            gradient = 2 * X.T.dot(errors) / m
            weights = weights - learning_rate * gradient

        elif method == "stochastic":
            # Update weights for each data point individually
            for i in range(m):
                prediction = X[i].dot(weights)
                error = prediction - y[i]
                gradient = 2 * X[i].T.dot(error)
                weights = weights - learning_rate * gradient

        elif method == "mini_batch":
            # Update weights using sequential batches of data points without shuffling
            for i in range(0, m, batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                predictions = X_batch.dot(weights)
                errors = predictions - y_batch
                gradient = 2 * X_batch.T.dot(errors) / batch_size
                weights = weights - learning_rate * gradient

    return weights
