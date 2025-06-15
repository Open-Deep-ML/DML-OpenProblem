import numpy as np

def to_categorical(x, n_col=None):
    # One-hot encoding of nominal values
    # If n_col is not provided, determine the number of columns from the input array
    if not n_col:
        n_col = np.amax(x) + 1
    # Initialize a matrix of zeros with shape (number of samples, n_col)
    one_hot = np.zeros((x.shape[0], n_col))
    # Set the appropriate elements to 1
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot
    
