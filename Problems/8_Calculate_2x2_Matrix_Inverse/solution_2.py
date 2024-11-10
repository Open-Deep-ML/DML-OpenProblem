import numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
	return np.linalg.inv(np.array(matrix))