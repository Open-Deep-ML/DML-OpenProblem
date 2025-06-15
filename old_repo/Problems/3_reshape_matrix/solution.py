import numpy as np

def reshape_matrix(a: list[list[int | float]],new_shape: tuple[int, int]) -> list[list[int | float]]:
    # Reshape the matrix using numpy and return it as a list of lists
    # Not compatible case
    if len(a)*len(a[0]) != new_shape[0]*new_shape[1]:
        return []
    # Compatible matches
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

    # Test case 4
    a = [[1, 2, 3, 4], [5, 6, 7, 8]]
    new_shape = (1, 4)
    assert reshape_matrix(a, new_shape) == []

    # Test case 5
    a = [[1, 2],[3, 4]]
    new_shape = (4, 1)
    assert reshape_matrix(a, new_shape) == [[1], [2], [3], [4]]

    # Test case 6
    a = [[1, 2, 3],[4, 5, 6]]
    new_shape = (2, 3)  
    assert reshape_matrix(a, new_shape) == [[1, 2, 3], [4, 5, 6]]

    # Test case 7
    a = [[1, 2, 3],[4, 5, 6]]
    new_shape = (6, 1)
    assert reshape_matrix(a, new_shape) == [[1], [2], [3], [4], [5], [6]]

    # Test case 8
    a = [[1, 2],[3, 4],[5, 6]]
    new_shape = (3, 2)  # same as original
    assert reshape_matrix(a, new_shape) == [[1, 2], [3, 4], [5, 6]]

    # Test case 9
    a = [[1.5, 2.2, 3.1],[4.7, 5.9, 6.3]]
    new_shape = (3, 2)
    assert reshape_matrix(a, new_shape) == [[1.5, 2.2],[3.1, 4.7],[5.9, 6.3]]

    # Test case 10
    a = [[1.5, 2.2, 3.1],[4.7, 5.9, 6.3],[7.7, 8.8, 9.9]]
    new_shape = (9, 1)
    assert reshape_matrix(a, new_shape) == [[1.5], [2.2], [3.1], [4.7], [5.9], [6.3], [7.7], [8.8], [9.9]]

    # Test case 11
    a = [[0, 0],[0, 0]]
    new_shape = (2, 2)
    assert reshape_matrix(a, new_shape) == [[0, 0], [0, 0]]

    # Test case 12
    a = [[1, 2]]
    new_shape = (2, 2)
    assert reshape_matrix(a, new_shape) == []

    # Test case 13
    a = [[10, 20, 30, 40]]
    new_shape = (2, 2)
    assert reshape_matrix(a, new_shape) == [[10, 20], [30, 40]]

    # Test case 14: 4x1 to 2x2
    a = [[1], [2], [3], [4]]
    new_shape = (2, 2)
    assert reshape_matrix(a, new_shape) == [[1, 2], [3, 4]]

    # Test case 15
    a = [[1], [2], [3], [4]]
    new_shape = (8, 1)
    assert reshape_matrix(a, new_shape) == []

    # Test case 16
    a = [[1], [2], [3], [4], [5]]
    new_shape = (1, 5)
    assert reshape_matrix(a, new_shape) == [[1, 2, 3, 4, 5]]

    # Test case 17
    a = [[-1, -2],[-3, -4]]
    new_shape = (1, 4)
    assert reshape_matrix(a, new_shape) == [[-1, -2, -3, -4]]

    # Test case 18
    a = [[-1, 2],[3, -4],[5, 6],[7, -8]]
    new_shape = (2, 4)
    assert reshape_matrix(a, new_shape) == [[-1, 2, 3, -4],[5, 6, 7, -8]]

    # Test case 19
    a = [[1, 2, 3, 4, 5, 6]]
    new_shape = (3, 2)
    assert reshape_matrix(a, new_shape) == [[1, 2], [3, 4], [5, 6]]

    # Test case 20
    a = [[1], [2], [3], [4], [5], [6]]
    new_shape = (2, 3)
    assert reshape_matrix(a, new_shape) == [[1, 2, 3], [4, 5, 6]]

if __name__ == "__main__":
    test_reshape_matrix()
    print("All reshape_matrix tests passed.")