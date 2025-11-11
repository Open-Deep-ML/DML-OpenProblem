def pad_matrix(a: list[list[int|float]], padding: int) -> list[list[int|float]]:
	# return a new matrix with padding of p zeros on both sides of each axis
	
	rows = len(a)
	cols = len(a[0])
	
	padded_matrix = [[0]*(cols + 2*padding) for _ in range(rows + 2*padding)]
	for i in range(rows):
		for j in range(cols):
			padded_matrix[i + padding][j + padding] = a[i][j]
			
	return padded_matrix