import numpy as np

def reshape_matrix(a: list[list[int | float]],new_shape: tuple[int, int]) -> list[list[int | float]]:
    # Reshape the matrix using numpy and return it as a list of lists
    return np.array(a).reshape(new_shape).tolist()

def test_reshape_matrix() -> None:
    # Test cases for reshape_matrix function

    # Test case 1
    a = [[1, 2, 3, 4], [5, 6, 7, 8]]
    new_shape = (4, 2)
    assert reshape_matrix(a, new_shape) == [[1, 2], [3, 4], [5, 6], [7, 8]]

    # Test case 2
    a = [[1, 2, 3], [4, 5, 6]]
    new_shape = (3, 2)
    assert reshape_matrix(a, new_shape) == [[1, 2], [3, 4], [5, 6]]

    # Test case 3
    a = [[1, 2, 3, 4], [5, 6, 7, 8]]
    new_shape = (2, 4)
    assert reshape_matrix(a, new_shape) == [[1, 2, 3, 4], [5, 6, 7, 8]]

if __name__ == "__main__":
    test_reshape_matrix()
    print("All reshape_matrix tests passed.")