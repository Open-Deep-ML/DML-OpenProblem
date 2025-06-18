def matrix_dot_vector(
    a: list[list[int | float]],
    b: list[int | float],
) -> list[int | float] | int:
    #Check that dimensions match
    if len(a[0]) != len(b):
	return -1
        
	c = []
    #Iterate using a list comprehension
	for s_a in a:
		temp = sum([s_a[i]*b[i] for i in range(len(b))])
		c.append(temp)
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

if __name__ == "__main__":
    test_matrix_dot_vector()
    print("All tests passed.")