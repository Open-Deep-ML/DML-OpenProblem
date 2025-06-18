import numpy as np

def rref(matrix):
    # Convert to float for division operations
    A = matrix.astype(np.float32)
    n, m = A.shape
    row = 0  # Current row index for pivot placement
    
    # Iterate over columns, up to the number of columns m
    for col in range(m):
        if row >= n:  # No more rows to process
            break
        
        # Find a row with a non-zero entry in the current column
        nonzero_rel_id = np.nonzero(A[row:, col])[0]
        if len(nonzero_rel_id) == 0:  # No pivot in this column
            continue
        
        # Swap the current row with the row containing the non-zero entry
        k = nonzero_rel_id[0] + row
        A[[row, k]] = A[[k, row]]
        
        # Normalize the pivot row to make the pivot 1
        A[row] = A[row] / A[row, col]
        
        # Eliminate all other entries in this column
        for j in range(n):
            if j != row:
                A[j] -= A[j, col] * A[row]
        
        row += 1  # Move to the next row for the next pivot
    
    return A
