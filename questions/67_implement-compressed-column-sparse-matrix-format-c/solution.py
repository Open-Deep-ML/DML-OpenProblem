def compressed_col_sparse_matrix(dense_matrix):
    vals = []
    row_idx = []
    col_ptr = [0]

    rows, cols = len(dense_matrix), len(dense_matrix[0])

    for i in range(cols):
        for j in range(rows):
            val = dense_matrix[j][i]
            if val != 0:
                vals.append(val)
                row_idx.append(j)
        col_ptr.append(len(vals))

    return vals, row_idx, col_ptr
