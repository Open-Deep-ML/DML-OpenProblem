import numpy as np
from scipy.sparse import csr_matrix

def compressed_row_sparse_matrix(dense_matrix: list[list[float]]) -> tuple[list[float], list[int], list[int]]:

    vals = []
    col_idx = []
    row_ptr = [0]

    for i, row in enumerate(dense_matrix):
        for j, val in enumerate(row):
            if val == 0:
                continue
            vals.append(val)
            col_idx.append(j)
        row_ptr.append(len(vals))   

    return vals, col_idx, row_ptr
    

def test_compressed_row():

    # Test case 1
    dense_matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    vals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)
    
    assert vals == [], "Test case 1 failed: vals should be an empty list"
    assert col_idx == [], "Test case 1 failed: col_idx should be an empty list"
    assert row_ptr == [0, 0, 0, 0], "Test case 1 failed: row_ptr should be [0, 0, 0, 0]"

    # Test case 2
    dense_matrix = [
        [0, 0, 0],
        [1, 2, 0],
        [0, 3, 4]
    ]

    vals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)
    
    assert vals == [1, 2, 3, 4], "Test case 2 failed: vals should be [1, 2, 3, 4]"
    assert col_idx == [0, 1, 1, 2], "Test case 2 failed: col_idx should be [0, 1, 1, 2]"
    assert row_ptr == [0, 0, 2, 4], "Test case 2 failed: row_ptr should be [0, 0, 2, 4]"

    # Test case 3
    dense_matrix = [
        [0, 0, 3, 0, 0],  # Only one non-zero element in the middle
        [0, 4, 0, 0, 0],  # One non-zero element
        [5, 0, 0, 6, 0],  # Two non-zero elements
        [0, 0, 0, 0, 0],  # All zeros
        [0, 7, 0, 0, 8]   # Two non-zero elements
    ]

    vals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)

    assert vals == [3, 4, 5, 6, 7, 8], "Test case failed: vals should be [3, 4, 5, 6, 7, 8]"
    assert col_idx == [2, 1, 0, 3, 1, 4], "Test case failed: col_idx should be [2, 1, 0, 3, 1, 4]"
    assert row_ptr == [0, 1, 2, 4, 4, 6], "Test case failed: row_ptr should be [0, 1, 2, 4, 4, 6]"

if __name__ == "__main__":
    test_compressed_row()
    print("All Compressed Row Tests passed.")
