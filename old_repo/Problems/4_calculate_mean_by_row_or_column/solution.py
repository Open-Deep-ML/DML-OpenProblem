def calculate_matrix_mean(
    matrix: list[list[float]],
    mode: str,
) -> list[float]:
    # Calculate mean by row or column
    if mode == 'column':
        return [sum(col) / len(matrix) for col in zip(*matrix)]
    elif mode == 'row':
        return [sum(row) / len(row) for row in matrix]
    else:
        raise ValueError("Mode must be 'row' or 'column'")

def test_calculate_matrix_mean() -> None:
    # Test cases for calculate_matrix_mean function

    # Test case 1
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mode = 'column'
    assert calculate_matrix_mean(matrix, mode) == [4.0, 5.0, 6.0]

    # Test case 2
    mode = 'row'
    assert calculate_matrix_mean(matrix, mode) == [2.0, 5.0, 8.0]

    # Test case 3
    matrix = [[1, 2], [3, 4], [5, 6]]
    mode = 'column'
    assert calculate_matrix_mean(matrix, mode) == [3.0, 4.0]

    # Test case 4
    mode = 'row'
    assert calculate_matrix_mean(matrix, mode) == [1.5, 3.5, 5.5]

if __name__ == "__main__":
    test_calculate_matrix_mean()
    print("All calculate_matrix_mean tests passed.")