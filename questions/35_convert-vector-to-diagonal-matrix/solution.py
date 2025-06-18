import numpy as np

def make_diagonal(x):
    identity_matrix = np.identity(np.size(x))
    return (identity_matrix*x)
    
