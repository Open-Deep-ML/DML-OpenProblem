def reshape_matrix(a: list[list[int | float]],new_shape: tuple[int, int]) -> list[list[int | float]]:
    # Extract dimensions
    rows,cols = len(a),len(a[0])
    new_rows, new_cols = new_shape
    # Check if reshaping is feasible
    if rows*cols != new_rows*new_cols:
        raise ValueError('Incompatible shapes')
    # Flatten matrix
    flat_matrix = []
    for r in range(rows):
        for c in range(cols):
            flat_matrix.append(a[r][c])
    # Populate new matrix
    reshaped_matrix = []
    # Use index to traverse all of the elements of the matrix in order
    index = 0
    for nr in range(new_rows):
        new_row = []
        for nc in range(new_cols):
            new_row.append(flat_matrix[index])
            index += 1
        reshaped_matrix.append(new_row)
    return reshaped_matrix
    

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