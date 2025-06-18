def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    if len(a) == 0:
        return []

    return [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]


def test_transpose_matrix() -> None:
    # empty/degenerate matrix
    assert transpose_matrix([]) == []
    assert transpose_matrix([[]]) == []

    # valid matrix
    a: list[list[int | float]] = [[1, 2, 3], [4, 5, 6]]
    assert transpose_matrix(a) == [[1, 4], [2, 5], [3, 6]]

if __name__ == "__main__":
    test_transpose_matrix()
    print("All tests passed.")