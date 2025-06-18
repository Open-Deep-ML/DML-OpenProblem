## Task: Convert a Dense Matrix to Compressed Row Sparse (CSR) Format

Your task is to implement a function that converts a given dense matrix into the Compressed Row Sparse (CSR) format, an efficient storage representation for sparse matrices. The CSR format only stores non-zero elements and their positions, significantly reducing memory usage for matrices with a large number of zeros.

Write a function `compressed_row_sparse_matrix(dense_matrix)` that takes a 2D list `dense_matrix` as input and returns a tuple containing three lists:

- **Values array**: List of all non-zero elements in row-major order.
- **Column indices array**: Column index for each non-zero element in the values array.
- **Row pointer array**: Cumulative number of non-zero elements per row, indicating the start of each row in the values array.

    
