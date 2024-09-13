import numpy as np

def transform_matrix(
    A: list[list[int | float]],
    T: list[list[int | float]],
    S: list[list[int | float]],
) -> list[list[int | float]]:
    
    # Convert to numpy arrays for easier manipulation
    A = np.array(A, dtype=float)
    T = np.array(T, dtype=float)
    S = np.array(S, dtype=float)
    
    # Check if the matrices T and S are invertible
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        # raise ValueError("The matrices T and/or S are not invertible.")
        return -1
    
    # Compute the inverse of T
    T_inv = np.linalg.inv(T)
    
    # Perform the matrix transformation; use @ for better readability
    transformed_matrix = np.round(T_inv @ A @ S, 3)
    
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
    assert transform_matrix(A, T, S) == [[-4.0, 2.0], [3.0, -1.0]]
    
    # Test case 3
    A = [[2, 3], [1, 4]]
    T = [[3, 0], [0, 3]]
    S = [[1, 1], [0, 1]]
    assert transform_matrix(A, T, S) == [[0.667, 1.667], [0.333, 1.667]]
    
    # Test case 4
    A = [[2, 3], [1, 4]]
    T = [[3, 0], [0, 3]]
    S = [[1, 1], [1, 1]]
    assert transform_matrix(A, T, S) == -1
    # try:
    #     transform_matrix(A, T, S)
    #     assert False, "ValueError not raised"
    # except ValueError as e:
    #     assert str(e) == "The matrices T and/or S are not invertible."

if __name__ == "__main__":
    test_transform_matrix()
    print("All transform_matrix tests passed.")