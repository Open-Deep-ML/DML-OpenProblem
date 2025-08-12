import numpy as np


def partial_pivoting(A_aug, row_num, col_num):
    rows, cols = A_aug.shape
    max_row = row_num
    max_val = abs(A_aug[row_num, col_num])
    for i in range(row_num, rows):
        current_val = abs(A_aug[i, col_num])
        if current_val > max_val:
            max_val = current_val
            max_row = i
    if max_row != row_num:
        A_aug[[row_num, max_row]] = A_aug[[max_row, row_num]]
    return A_aug


def gaussian_elimination(A, b):
    rows, cols = A.shape
    A_aug = np.hstack((A, b.reshape(-1, 1)))

    for i in range(rows - 1):
        A_aug = partial_pivoting(A_aug, i, i)
        for j in range(i + 1, rows):
            A_aug[j, i:] -= (A_aug[j, i] / A_aug[i, i]) * A_aug[i, i:]

    x = np.zeros_like(b, dtype=float)
    for i in range(rows - 1, -1, -1):
        x[i] = (A_aug[i, -1] - np.dot(A_aug[i, i + 1 : cols], x[i + 1 :])) / A_aug[i, i]
    return x
