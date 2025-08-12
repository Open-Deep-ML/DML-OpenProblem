def matrix_dot_vector(
    a: list[list[int | float]], b: list[int | float]
) -> list[int | float]:
    # if the no. of columns of a is not equal to the len of vector b then return -1
    if len(a[0]) != len(b):
        return -1
        # create a vector to store the values
    result = []
    # traverse through the matrix and add do the dot product and append to the result vector
    for i in range(len(a)):
        val = 0
        for j in range(len(b)):
            val += a[i][j] * b[j]
        result.append(val)
    return result
