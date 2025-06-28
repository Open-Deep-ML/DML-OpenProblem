import numpy as np


def train_simple_cnn_with_backprop(
    X, y, epochs, learning_rate, kernel_size=3, num_filters=1
):
    """
    Trains a simple CNN with one convolutional layer, ReLU activation, flattening, and a dense layer with softmax output using backpropagation.

    Assumes X has shape (n_samples, height, width) for grayscale images and y is one-hot encoded with shape (n_samples, num_classes).

    Parameters:
    X : np.ndarray, input data
    y : np.ndarray, one-hot encoded labels
    epochs : int, number of training epochs
    learning_rate : float, learning rate for weight updates
    kernel_size : int, size of the square convolutional kernel
    num_filters : int, number of filters in the convolutional layer

    Returns:
    W_conv, b_conv, W_dense, b_dense : Trained weights and biases for the convolutional and dense layers
    """
    n_samples, height, width = X.shape
    num_classes = y.shape[1]

    # Initialize weights and biases
    W_conv = np.random.randn(kernel_size, kernel_size, num_filters) * 0.01
    b_conv = np.zeros(num_filters)
    output_height = height - kernel_size + 1
    output_width = width - kernel_size + 1
    flattened_size = output_height * output_width * num_filters
    W_dense = np.random.randn(flattened_size, num_classes) * 0.01
    b_dense = np.zeros(num_classes)

    for epoch in range(epochs):
        for i in range(n_samples):  # Stochastic Gradient Descent with batch size 1
            # Forward pass
            # Convolutional layer
            Z_conv = np.zeros((output_height, output_width, num_filters))
            for k in range(num_filters):
                for p in range(output_height):
                    for q in range(output_width):
                        Z_conv[p, q, k] = (
                            np.sum(
                                X[i, p : p + kernel_size, q : q + kernel_size]
                                * W_conv[:, :, k]
                            )
                            + b_conv[k]
                        )
            A_conv = np.maximum(Z_conv, 0)  # ReLU activation
            A_flat = A_conv.flatten()  # Flatten the output

            # Dense layer
            Z_dense = np.dot(A_flat, W_dense) + b_dense
            exp_Z_dense = np.exp(
                Z_dense - np.max(Z_dense)
            )  # Numerical stability for softmax
            A_dense = exp_Z_dense / np.sum(exp_Z_dense)

            # Backpropagation
            # Loss gradient for cross-entropy with softmax
            dZ_dense = A_dense - y[i]

            # Dense layer gradients
            dW_dense = np.outer(A_flat, dZ_dense)
            db_dense = dZ_dense
            dA_flat = np.dot(dZ_dense, W_dense.T)

            # Reshape and backprop through ReLU
            dA_conv = dA_flat.reshape(A_conv.shape)
            dZ_conv = dA_conv * (A_conv > 0).astype(float)

            # Convolutional layer gradients
            dW_conv = np.zeros_like(W_conv)
            db_conv = np.zeros(num_filters)
            for k in range(num_filters):
                db_conv[k] = np.sum(dZ_conv[:, :, k])
                for ii in range(kernel_size):
                    for jj in range(kernel_size):
                        dW_conv[ii, jj, k] = np.sum(
                            dZ_conv[:, :, k]
                            * X[i, ii : ii + output_height, jj : jj + output_width]
                        )

            # Update weights and biases
            W_conv -= learning_rate * dW_conv
            b_conv -= learning_rate * db_conv
            W_dense -= learning_rate * dW_dense
            b_dense -= learning_rate * db_dense

    return W_conv, b_conv, W_dense, b_dense
