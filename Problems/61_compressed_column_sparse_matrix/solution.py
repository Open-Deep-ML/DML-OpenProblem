import numpy as np
from scipy.sparse import csc_matrix

def compressed_col_sparse_matrix(dense_matrix: list[list[float]]) -> tuple[list[float], list[int], list[int]]:

    vals = []
    row_idx = []
    col_ptr = [0]

    rows, cols = len(dense_matrix), len(dense_matrix[0])

    for i in range(cols):
        for j in range(rows):
            val = dense_matrix[j][i]
            if val == 0:
                continue
            vals.append(val)
            row_idx.append(j)
        col_ptr.append(len(vals))   

    return vals, row_idx, col_ptr
    

def test_compressed_col():

    # Test case 1
    dense_matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    vals, row_idx, col_ptr = compressed_col_sparse_matrix(dense_matrix)
    
    assert vals == [], "Test case 1 failed: vals should be an empty list"
    assert row_idx == [], "Test case 1 failed: row_idx should be an empty list"
    assert col_ptr == [0, 0, 0, 0], "Test case 1 failed: col_ptr should be [0, 0, 0, 0]"

    # Test case 2
    dense_matrix = [
        [0, 0, 0],
        [1, 2, 0],
        [0, 3, 4]
    ]

    vals, row_idx, col_ptr = compressed_col_sparse_matrix(dense_matrix)
    
    assert vals == [1, 2, 3, 4], "Test case 2 failed: vals should be [1, 2, 3, 4]"
    assert row_idx == [1, 1, 2, 2], "Test case 2 failed: row_idx should be [1, 1, 2, 2]"
    assert col_ptr == [0, 1, 3, 4], "Test case 2 failed: col_ptr should be [0, 1, 3, 4]"

    # Test case 3
    dense_matrix = [
        [0, 0, 3, 0, 0],
        [0, 4, 0, 0, 0], 
        [5, 0, 0, 6, 0],
        [0, 0, 0, 0, 0], 
        [0, 7, 0, 0, 8] 
    ]

    vals, row_idx, col_ptr = compressed_col_sparse_matrix(dense_matrix)

    assert vals == [5, 4, 7, 3, 6, 8], "Test case failed: vals should be [5, 4, 7, 3, 6, 8]"
    assert row_idx == [2, 1, 4, 0, 2, 4], "Test case failed: row_idx should be [2, 1, 4, 0, 2, 4]"
    assert col_ptr == [0, 1, 3, 4, 5, 6], "Test case failed: col_ptr should be [0, 1, 3, 4, 5, 6]"

if __name__ == "__main__":
    test_compressed_row()
    print("All Compressed Column Tests passed.")
