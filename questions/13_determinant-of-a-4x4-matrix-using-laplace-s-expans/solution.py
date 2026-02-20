def determinant_4x4(matrix: list[list[int | float]]) -> float:
    # Base case: If the matrix is 1x1, return its single element
    if len(matrix) == 1:
        return matrix[0][0]
    # Recursive case: Calculate determinant using Laplace's Expansion
    det = 0
    for c in range(len(matrix)):
        minor = [row[:c] + row[c + 1 :] for row in matrix[1:]]  # Remove column c
        cofactor = ((-1) ** c) * determinant_4x4(minor)  # Compute cofactor
        det += matrix[0][c] * cofactor  # Add to running total
    return det
