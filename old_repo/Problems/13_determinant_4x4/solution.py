def determinant_4x4(matrix: list[list[int|float]]) -> float:
    # Base case: If the matrix is 1x1, return its single element
    if len(matrix) == 1:
        return matrix[0][0]
    # Recursive case: Calculate determinant using Laplace's Expansion
    det = 0
    for c in range(len(matrix)):
        minor = [row[:c] + row[c+1:] for row in matrix[1:]]  # Remove column c
        cofactor = ((-1)**c) * determinant_4x4(minor)  # Compute cofactor
        det += matrix[0][c] * cofactor  # Add to running total
    return det

def test_determinant_4x4() -> None:
    # Test case 1
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    assert determinant_4x4(matrix) == 0, "Test case 1 failed"

    # Test case 2
    matrix = [[4, 3, 2, 1], [3, 2, 1, 4], [2, 1, 4, 3], [1, 4, 3, 2]]
    assert determinant_4x4(matrix) == -160, "Test case 2 failed"

    # Test case 3
    matrix = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
    assert determinant_4x4(matrix) == 0, "Test case 3 failed"

if __name__ == "__main__":
    test_determinant_4x4()
    print("All determinant_4x4 tests passed.")
