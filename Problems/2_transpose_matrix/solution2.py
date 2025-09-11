def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:

    # this is a quick and efficient way to transpose a matrix using
    # the built-in *argument unpacking and the zip function.

    # The flow of the input matrix `a` goes as follows:
    # Input: [[1, 2, 3], [4, 5, 6], [3, 2, 7]] (this is *a)
    # zip(*a) produces:
    # (1, 4, 3), (2, 5, 2), (3, 6, 7)
    # finally, we convert these tuples to lists and return the result.

    return [list(row) for row in zip(*a)]


if __name__ == "__main__":
    a = [[1, 2, 3], [4, 5, 6], [3, 2, 7]]
    transposed = transpose_matrix(a)

    assert transposed == [
        [1, 4, 3],
        [2, 5, 2],
        [3, 6, 7],
    ], (
        f"Test failed: Transposed matrix does not match expected output.\n"
        f"Expected:\n{[1, 4, 3], [2, 5, 2], [3, 6, 7]}\n"
        f"Got:\n{transposed}"
    )
    print(
        f"Test passed: Transposed matrix matches expected output.\n"
        f"Input matrix:\n{a}\n"
        f"Transposed matrix:\n{transposed}"
    )
