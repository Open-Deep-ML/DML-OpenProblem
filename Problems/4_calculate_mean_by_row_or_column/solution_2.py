def calculate_matrix_mean(
    matrix: list[list[float]],
    mode: str,
) -> list[float]:
    # Calculate mean by row or column
    if mode not in ['row','column']:
        raise ValueError("Mode has to be row or column")
    #Get dimensions and initialize return array
    rows, columns = len(matrix), len(matrix[0])
    means = []
    #Row mode
    if mode == 'row':
        for r in range(rows):
            mean = 0
            for c in range(columns):
                mean += matrix[r][c]
            mean /= columns
            means.append(mean)
    #Column mode
    else:
       for c in range(columns):
            mean = 0
            for r in range(rows):
                mean += matrix[r][c]
            mean /= rows
            means.append(mean) 
    #Return
    return means

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

    # Test case 5
    matrix = [[-1, -2], [-3, -4]]
    mode = 'row'
    assert calculate_matrix_mean(matrix, mode) == [-1.5, -3.5]

    # Test case 6
    mode = 'column'
    assert calculate_matrix_mean(matrix, mode) == [-2.0, -3.0]

    # Test case 7
    matrix = [[1.5, 2.5, 3.5]]
    mode = 'row'
    assert calculate_matrix_mean(matrix, mode) == [2.5]

    # Test case 8
    mode = 'column'
    assert calculate_matrix_mean(matrix, mode) == [1.5, 2.5, 3.5]

    # Test case 9
    matrix = [[10, 20, 30]]
    mode = 'row'
    assert calculate_matrix_mean(matrix, mode) == [20.0]

    # Test case 10
    mode = 'column'
    assert calculate_matrix_mean(matrix, mode) == [10.0, 20.0, 30.0]

    # Test case 11
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8]]
    mode = 'row'
    assert calculate_matrix_mean(matrix, mode) == [2.5, 6.5]

    # Test case 12
    mode = 'column'
    assert calculate_matrix_mean(matrix, mode) == [3.0, 4.0, 5.0, 6.0]

    # Test case 13
    matrix = [[2, 2], [2, 2]]
    mode = 'row'
    assert calculate_matrix_mean(matrix, mode) == [2.0, 2.0]

    # Test case 14
    mode = 'column'
    assert calculate_matrix_mean(matrix, mode) == [2.0, 2.0]

    # Test case 15
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    mode = 'row'
    assert calculate_matrix_mean(matrix, mode) == [2.0, 5.0, 8.0, 11.0]

    # Test case 16
    mode = 'column'
    assert calculate_matrix_mean(matrix, mode) == [5.5, 6.5, 7.5]

    # Test case 17
    matrix = [[1, 2], [3, 4]]
    mode = 'rows'
    try:
        calculate_matrix_mean(matrix, mode)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 18
    mode = 'columns'
    try:
        calculate_matrix_mean(matrix, mode)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 19
    mode = 'average'
    try:
        calculate_matrix_mean(matrix, mode)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 20
    mode = ''
    try:
        calculate_matrix_mean(matrix, mode)
        assert False, "Expected ValueError"
    except ValueError:
        pass

if __name__ == "__main__":
    test_calculate_matrix_mean()
    print("All calculate_matrix_mean tests passed.")