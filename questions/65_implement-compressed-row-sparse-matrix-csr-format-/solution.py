import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
    vals = []
    col_idx = []
    row_ptr = [0]

    for row in dense_matrix:
        for j, val in enumerate(row):
            if val != 0:
                vals.append(val)
                col_idx.append(j)
        row_ptr.append(len(vals))

    return vals, col_idx, row_ptr
