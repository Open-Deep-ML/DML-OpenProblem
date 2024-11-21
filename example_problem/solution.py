def matrix_dot_vector(
    a: list[list[int | float]],
    b: list[int | float],
) -> list[int | float] | int:
    # verify MxN matrix is compatible with N vector
    if (len(a[0]) if a else 0) != len(b):
        return -1

    # perform the multiplication
    c: list[int | float] = [0] * len(a)
    for i in range(len(a)):
        for j in range(len(a[i])):
            c[i] += a[i][j] * b[j]

    return c


def test_matrix_dot_vector() -> None:
    # empty product
    assert matrix_dot_vector([], []) == []

    # invalid product
    assert matrix_dot_vector([], [1, 2]) == -1
    assert matrix_dot_vector([[1, 2]], []) == -1
    assert matrix_dot_vector([[1, 2], [2, 4]], [1]) == -1

    # valid product
    a: list[list[int | float]] = [[1, 2], [2, 4]]
    b: list[int | float] = [1, 2]
    assert matrix_dot_vector(a, b) == [5, 10]
    
    # valid product with rectangular matrix (non-square) -- > if we changed  "for j in range(len(a[i]))" to "for j in range(len(a))" previous tests will pass 
    a: list[list[int | float]] = [[1, 2, 3], [2, 4, 6]]
    b: list[int | float] = [1, 2, 3]
    assert matrix_dot_vector(a, b) == [14, 28]

if __name__ == "__main__":
    test_matrix_dot_vector()
    print("All tests passed.")
