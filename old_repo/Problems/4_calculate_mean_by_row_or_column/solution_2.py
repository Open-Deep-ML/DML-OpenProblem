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

if __name__ == "__main__":
    test_calculate_matrix_mean()
    print("All calculate_matrix_mean tests passed.")