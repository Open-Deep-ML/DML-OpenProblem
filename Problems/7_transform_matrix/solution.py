import numpy as np

def transform_matrix(
    A: list[list[int | float]],
    T: list[list[int | float]],
    S: list[list[int | float]],
) -> list[list[int | float]]:
    # Convert to numpy arrays for easier manipulation
    A = np.array(A)
    T = np.array(T)
    S = np.array(S)
    
    # Check if the matrices T and S are invertible
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        raise ValueError("The matrices T and/or S are not invertible.")
    
    # Compute the inverses of T and S
    T_inv = np.linalg.inv(T)
    
    # Perform the matrix transformation
    transformed_matrix = np.round(T_inv.dot(A).dot(S),3)
    
    return transformed_matrix.tolist()

def test_transform_matrix() -> None:
    # Test cases for transform_matrix function

    # Test case 1
    A = [[1, 2], [3, 4]]
    T = [[2, 0], [0, 2]]
    S = [[1, 1], [0, 1]]
    assert transform_matrix(A, T, S) == [[0.5, 1.5], [1.5, 3.5]]

    # Test case 2
    A = [[1, 0], [0, 1]]
    T = [[1, 2], [3, 4]]
    S = [[2, 0], [0, 2]]
    assert transform_matrix(A, T, S) == [[-4., 2.], [3., -1.]]

    # Test case 3
    A = [[2, 3], [1, 4]]
    T = [[3, 0], [0, 3]]
    S = [[1, 1], [0, 1]]
    print(transform_matrix(A, T, S))

    assert transform_matrix(A, T, S) == [[0.667, 1.667], [0.333, 1.667]]

if __name__ == "__main__":
    test_transform_matrix()
    print("All transform_matrix tests passed.")