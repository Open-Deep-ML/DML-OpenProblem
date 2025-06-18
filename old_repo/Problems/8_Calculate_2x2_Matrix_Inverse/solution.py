def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    determinant = a * d - b * c
    if determinant == 0:
        return None
    inverse = [[d/determinant, -b/determinant], [-c/determinant, a/determinant]]
    return inverse

def test_inverse_2x2() -> None:
    # Test cases for inverse_2x2 function

    # Test case 1
    matrix = [[4, 7], [2, 6]]
    assert inverse_2x2(matrix) == [[0.6, -0.7], [-0.2, 0.4]]

    # Test case 2
    matrix = [[2, 1], [6, 2]]
    assert inverse_2x2(matrix) == [[-1.0, 0.5], [3.0, -1.0]]

if __name__ == "__main__":
    test_inverse_2x2()
    print("All inverse_2x2 tests passed.")
