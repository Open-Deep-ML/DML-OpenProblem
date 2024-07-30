def calculate_eigenvalues(
    matrix: list[list[float]],
) -> list[float]:
    # Calculate eigenvalues for a 2x2 matrix
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    trace = a + d
    determinant = a * d - b * c
    # Calculate the discriminant of the quadratic equation
    discriminant = trace**2 - 4 * determinant
    # Solve for eigenvalues
    lambda_1 = (trace + discriminant**0.5) / 2
    lambda_2 = (trace - discriminant**0.5) / 2
    return [lambda_1, lambda_2]

def test_calculate_eigenvalues() -> None:
    # Test cases for calculate_eigenvalues function

    # Test case 1
    matrix = [[2, 1], [1, 2]]
    assert calculate_eigenvalues(matrix) == [3.0, 1.0]

    # Test case 2
    matrix = [[4, -2], [1, 1]]
    assert calculate_eigenvalues(matrix) == [3.0, 2.0]

if __name__ == "__main__":
    test_calculate_eigenvalues()
    print("All calculate_eigenvalues tests passed.")