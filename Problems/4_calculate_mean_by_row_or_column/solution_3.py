def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    means = []
    res = 0
    if mode == 'column':
        for i in range (0,len(matrix[0])):
            for j in range (0,len(matrix)):
                res = res + matrix[j][i]
            res = res / len(matrix)
            means.append(res)
            res = 0
    if mode == 'row':
        for i in range (0,len(matrix)):
            for j in range (0,len(matrix[0])):
                res = res + matrix[i][j]
            res = res / len(matrix[0])
            means.append(res)
            res = 0
	return means
