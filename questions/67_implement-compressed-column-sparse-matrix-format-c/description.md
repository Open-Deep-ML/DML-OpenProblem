## Task: Create a Compressed Column Sparse Matrix Representation

Your task is to implement a function that converts a dense matrix into its Compressed Column Sparse (CSC) representation. The CSC format stores only non-zero elements of the matrix and is efficient for matrices with a high number of zero elements.

Write a function `compressed_col_sparse_matrix(dense_matrix)` that takes in a two-dimensional list `dense_matrix` and returns a tuple of three lists:

- `values`: List of non-zero elements, stored in column-major order.
- `row indices`: List of row indices corresponding to each value in the values array.
- `column pointer`: List that indicates the starting index of each column in the values array.

    
