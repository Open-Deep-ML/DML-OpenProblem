import numpy as np

def rref(A):
    # Convert to float for division operations
    A = A.astype(np.float32)
    n, m = A.shape

    for i in range(n):
        if A[i,i] == 0:
            nonzero_current_row = np.nonzero(A[i:,i])[0] + i
            if len(nonzero_current_row) == 0:
                continue
            A[[i,nonzero_current_row[0]]] = A[[nonzero_current_row[0], i]]
        
        A[i] = A[i] / A[i,i]

        for j in range(n):
            if i!=j:
                A[j] -= A[i] * A[j,i]
    return A

def find_pivot_columns(A):

    n, m = A.shape

    pivot_columns = []

    for i in range(n):
        nonzero = np.nonzero(A[i,:])[0]

        if len(nonzero) != 0:
            pivot_columns.append(nonzero[0])
        
    return pivot_columns


def matrix_image(A):

    # find reduced row echelon form of matrix
    Arref = rref(A)

    # find pivots - ie the basis of the matrix image
    pivot_columns = find_pivot_columns(Arref)

    # get cols with pivot columns - ie the basis defining the matrix image
    image_basis = A[:, pivot_columns]

    return image_basis

def test_matrix_image() -> None:
    # Test cases for matrix_image function

    # Test case 1: Simple 2x2 identity matrix
    matrix = np.array([[1, 0], [0, 1]])
    image = matrix_image(matrix)
    expected_result = [[1, 0], [0, 1]]  # Column space should be the identity matrix
    assert np.array_equal(image, expected_result), f"Expected {expected_result}, but got {image}"

    # Test case 2: Matrix with linearly dependent columns
    matrix = np.array([[1, 2], [2, 4]])
    image = matrix_image(matrix)
    expected_result = [[1], [2]]  # Only one independent column
    assert np.array_equal(image, expected_result), f"Expected {expected_result}, but got {image}"

    # Test case 3: 3x3 matrix with a mix of dependent and independent columns
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    image = matrix_image(matrix)
    expected_result = [[1, 2], [4, 5], [7, 8]]  # Two independent columns
    assert np.array_equal(image, expected_result), f"Expected {expected_result}, but got {image}"

if __name__ == "__main__":
    test_matrix_image()
    print("All tests passed.")