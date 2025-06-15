
import numpy as np

def rref(A):
    # Convert to float for division operations
    A = A.astype(np.float32)
    n, m = A.shape

    for i in range(n):
        if A[i, i] == 0:
            nonzero_current_row = np.nonzero(A[i:, i])[0] + i
            if len(nonzero_current_row) == 0:
                continue
            A[[i, nonzero_current_row[0]]] = A[[nonzero_current_row[0], i]]

        A[i] = A[i] / A[i, i]

        for j in range(n):
            if i != j:
                A[j] -= A[i] * A[j, i]
    return A

def find_pivot_columns(A):
    n, m = A.shape
    pivot_columns = []
    for i in range(n):
        nonzero = np.nonzero(A[i, :])[0]
        if len(nonzero) != 0:
            pivot_columns.append(nonzero[0])
    return pivot_columns

def matrix_image(A):
    # Find the RREF of the matrix
    Arref = rref(A)
    # Find the pivot columns
    pivot_columns = find_pivot_columns(Arref)
    # Extract the pivot columns from the original matrix
    image_basis = A[:, pivot_columns]
    return image_basis
