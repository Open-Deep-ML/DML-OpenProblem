problems = {
    1: {
        'id': 1,
        'title': 'Matrix times Vector (easy)',
        'description': "Write a Python function that takes the dot product of a matrix and a vector. return -1 if the matrix could not be dotted with the vector",
        'example': """Example:
        input: a = [[1,2],[2,4]], b = [1,2]
        output:[5, 10] 
        reasoning: 1*1 + 2*2 = 5;
                   1*2+ 2*4 = 10""",
        'video':"https://youtu.be/DNoLs5tTGAw?si=vpkPobZMA8YY10WY",
        'learn': r'''
<h2>Matrix Times Vector</h2>

Consider a matrix \(A\) and a vector \(v\), where:

Matrix \(A\):
\[
A = \begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{pmatrix}
\]

Vector \(v\):
\[
v = \begin{pmatrix}
v_1 \\
v_2
\end{pmatrix}
\]

The dot product of \(A\) and \(v\) results in a new vector:
\[
A \cdot v = \begin{pmatrix}
a_{11}v_1 + a_{12}v_2 \\
a_{21}v_1 + a_{22}v_2
\end{pmatrix}
\]

Things to note: an \(n \times m\) matrix will need to be multiplied by a vector of size \(m\) or else this will not work.
''',
        'starter_code': "def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:\n    return c",
        'solution': """def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
    if len(a[0]) != len(b):
        return -1
    vals = []
    for i in a:
        hold = 0
        for j in range(len(i)):
            hold+=(i[j] * b[j])
        vals.append(hold)

    return vals""",
        'test_cases': [
            {"test": "print(matrix_dot_vector([[1,2,3],[2,4,5],[6,8,9]],[1,2,3]))", "expected_output": "[14, 25, 49]"},
            {"test": "print(matrix_dot_vector([[1,2],[2,4],[6,8],[12,4]],[1,2,3]))", "expected_output": "-1"},
        ],
    },
    2: {
        'id': 2,
        'title': 'Transpose of a Matrix (easy)',
        'description': "Write a Python function that computes the transpose of a given matrix.",
        'example': """Example:
        input: a = [[1,2,3],[4,5,6]]
        output: [[1,4],[2,5],[3,6]]
        reasoning: The transpose of a matrix is obtained by flipping rows and columns.""",
        'video': "https://youtu.be/fj0ZJ2gTSmI?si=vG8VSqASjyG0eNLY",
        'learn': r'''
<h2>Transpose of a Matrix</h2>

Consider a matrix \(M\) and its transpose \(M^T\), where:

Original Matrix \(M\):
\[
M = \begin{pmatrix} 
a & b & c \\ 
d & e & f 
\end{pmatrix}
\]

Transposed Matrix \(M^T\):
\[
M^T = \begin{pmatrix} 
a & d \\ 
b & e \\ 
c & f 
\end{pmatrix}
\]

Transposing a matrix involves converting its rows into columns and vice versa. This operation is fundamental in linear algebra for various computations and transformations.
''',
        'starter_code': "def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:\n    return b",
        'solution': """def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    return [list(i) for i in zip(*a)]""",
        'test_cases': [
            {"test": "print(transpose_matrix([[1,2],[3,4],[5,6]]))", "expected_output": "[[1, 3, 5], [2, 4, 6]]"},
            {"test": "print(transpose_matrix([[1,2,3],[4,5,6]]))", "expected_output": "[[1, 4], [2, 5], [3, 6]]"}
        ]
    },
    3: {
        'id': 3,
        'title': 'Reshape Matrix (easy)',
        'description': "Write a Python function that reshapes a given matrix into a specified shape.",
        'example': """Example:
        input: a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)
        output: [[1, 2], [3, 4], [5, 6], [7, 8]]
        reasoning: The given matrix is reshaped from 2x4 to 4x2.""",
        'video': "Coming Soon",
        'learn': r'''
<h2>Reshaping a Matrix</h2>

Matrix reshaping involves changing the shape of a matrix without altering its data. This is essential in many machine learning tasks where the input data needs to be formatted in a specific way.

For example, consider a matrix \(M\):

Original Matrix \(M\):
\[
M = \begin{pmatrix} 
1 & 2 & 3 & 4 \\ 
5 & 6 & 7 & 8 
\end{pmatrix}
\]

Reshaped Matrix \(M'\) with shape (4, 2):
\[
M' = \begin{pmatrix} 
1 & 2 \\ 
3 & 4 \\ 
5 & 6 \\ 
7 & 8 
\end{pmatrix}
\]

Ensure the total number of elements remains constant during reshaping.
''',
        'starter_code': "import numpy as np\n\ndef reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:\n    #Write your code here and return a python list after reshaping by using numpy's tolist() method\n    return reshaped_matrix",
        'solution': """def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int|float]) -> list[list[int|float]]:
    return np.array(a).reshape(new_shape).tolist()""",
        'test_cases': [
            {"test": "print(reshape_matrix([[1,2,3,4],[5,6,7,8]], (4, 2)))", "expected_output": "[[1, 2], [3, 4], [5, 6], [7, 8]]"},
            {"test": "print(reshape_matrix([[1,2,3],[4,5,6]], (3, 2)))", "expected_output": "[[1, 2], [3, 4], [5, 6]]"},
            {"test": "print(reshape_matrix([[1,2,3,4],[5,6,7,8]], (2, 4)))", "expected_output": "[[1, 2, 3, 4], [5, 6, 7, 8]]"},
        ],
    },
    4: {
        'id': 4,
        'title': 'Calculate Mean by Row or Column (easy)',
        'description': "Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.",
        'example': """Example1:
        input: matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'
        output: [4.0, 5.0, 6.0]
        reasoning: Calculating the mean of each column results in [(1+4+7)/3, (2+5+8)/3, (3+6+9)/3].
        
        Example 2:
        input: matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'row'
        output: [2.0, 5.0, 8.0]
        reasoning: Calculating the mean of each row results in [(1+2+3)/3, (4+5+6)/3, (7+8+9)/3].""",
        'video':'Coming Soon',
        'learn': r'''
<h2>Calculate Mean by Row or Column</h2>

Calculating the mean of a matrix by row or column involves averaging the elements across the specified dimension. This operation provides insights into the distribution of values within the dataset, useful for data normalization and scaling.

<h3> Row Mean </h3>
The mean of a row is computed by summing all elements in the row and dividing by the number of elements. For row \(i\), the mean is:
\[
\mu_{\text{row } i} = \frac{1}{n} \sum_{j=1}^{n} a_{ij}
\]
where \(a_{ij}\) is the matrix element in the \(i^{th}\) row and \(j^{th}\) column, and \(n\) is the total number of columns.

<h3> Column Mean </h3>
Similarly, the mean of a column is found by summing all elements in the column and dividing by the number of elements. For column \(j\), the mean is:
\[
\mu_{\text{column } j} = \frac{1}{m} \sum_{i=1}^{m} a_{ij}
\]
where \(m\) is the total number of rows.

This mathematical formulation helps in understanding how data is aggregated across different dimensions, a critical step in various data preprocessing techniques.
''',
        'starter_code': "def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:\n    return means",
        'solution': """def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == 'column':
        return [sum(col) / len(matrix) for col in zip(*matrix)]
    elif mode == 'row':
        return [sum(row) / len(row) for row in matrix]
    else:
        raise ValueError("Mode must be 'row' or 'column'")""",
        'test_cases': [
            {
                "test": "print(calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'column'))",
                "expected_output": "[4.0, 5.0, 6.0]"
            },
            {
                "test": "print(calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'row'))",
                "expected_output": "[2.0, 5.0, 8.0]"
            }
        ]
    },
    5: {
        'id': 5,
        'title': 'Scalar Multiplication of a Matrix (easy)',
        'description': "Write a Python function that multiplies a matrix by a scalar and returns the result.",
        'example': """Example:
        input: matrix = [[1, 2], [3, 4]], scalar = 2
        output: [[2, 4], [6, 8]]
        reasoning: Each element of the matrix is multiplied by the scalar.""",
        'video': "https://youtu.be/iE2NvpvZRBk",
        'learn': r'''
<h2>Scalar Multiplication of a Matrix</h2>

When a matrix \(A\) is multiplied by a scalar \(k\), the operation is defined as multiplying each element of \(A\) by \(k\). 

Given a matrix \(A\):
\[
A = \begin{pmatrix} 
a_{11} & a_{12} \\ 
a_{21} & a_{22} 
\end{pmatrix}
\]

And a scalar \(k\), the result of the scalar multiplication \(kA\) is:
\[
kA = \begin{pmatrix} 
ka_{11} & ka_{12} \\ 
ka_{21} & ka_{22} 
\end{pmatrix}
\]

This operation scales the matrix by \(k\) without changing its dimension or the relative proportion of its elements.
''',
        'starter_code': "def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:\n    return result",
        'solution': """def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    return [[element * scalar for element in row] for row in matrix]""",
        'test_cases': [
            {"test": "print(scalar_multiply([[1,2],[3,4]], 2))", "expected_output": "[[2, 4], [6, 8]]"},
            {"test": "print(scalar_multiply([[0,-1],[1,0]], -1))", "expected_output": "[[0, 1], [-1, 0]]"}
        ]
    },
    6: {
        'id': 6,
        'title': 'Calculate Eigenvalues of a Matrix (medium)',
        'description': "Write a Python function that calculates the eigenvalues of a 2x2 matrix. The function should return a list containing the eigenvalues, sort values from highest to lowest.",
        'example': """Example:
        input: matrix = [[2, 1], [1, 2]]
        output: [3.0, 1.0]
        reasoning: The eigenvalues of the matrix are calculated using the characteristic equation of the matrix, which for a 2x2 matrix is $\lambda^2 - \text{trace}(A)\lambda + \text{det}(A) = 0$, where $\lambda$ are the eigenvalues.""",
        'video': 'https://youtu.be/AMCFzIaHc4Y',
        'learn': r'''
<h2>Calculate Eigenvalues</h2>

Eigenvalues of a matrix offer significant insight into the matrix's behavior, particularly in the context of linear transformations and systems of linear equations.

<h3> Definition </h3>
For a square matrix \(A\), eigenvalues are scalars \(\lambda\) that satisfy the equation for some non-zero vector \(v\) (eigenvector):
\[
Av = \lambda v
\]

<h3> Calculation for a 2x2 Matrix </h3>
The eigenvalues of a 2x2 matrix \(A\), given by:
\[
A = \begin{pmatrix} 
a & b \\ 
c & d 
\end{pmatrix}
\]
are determined by solving the characteristic equation:
\[
\det(A - \lambda I) = 0
\]
This simplifies to a quadratic equation:
\[
\lambda^2 - \text{tr}(A) \lambda + \det(A) = 0
\]
Here, the trace of A, denoted as tr(A), is \(a + d\), and the determinant of A, denoted as det(A), is \(ad - bc\). Solving this equation yields the eigenvalues, \(\lambda\).

<h3> Significance </h3>
Understanding eigenvalues is essential for analyzing the effects of linear transformations represented by the matrix. They are crucial in various applications, including stability analysis, vibration analysis, and Principal Component Analysis (PCA) in machine learning.
''',
        'starter_code': "def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:\n    return eigenvalues",
        'solution': """def calculate_eigenvalues(matrix: list[list[float]]) -> list[float]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    trace = a + d
    determinant = a * d - b * c
    # Calculate the discriminant of the quadratic equation
    discriminant = trace**2 - 4 * determinant
    # Solve for eigenvalues
    lambda_1 = (trace + discriminant**0.5) / 2
    lambda_2 = (trace - discriminant**0.5) / 2
    return [lambda_1, lambda_2]""",
        'test_cases': [
            {
                "test": "print(calculate_eigenvalues([[2, 1], [1, 2]]))",
                "expected_output": "[3.0, 1.0]"
            },
            {
                "test": "print(calculate_eigenvalues([[4, -2], [1, 1]]))",
                "expected_output": "[3.0, 2.0]"
            }
        ]
    },
    7: {
        'id': 7,
        'title': 'Matrix Transformation (medium)',
        'description': "Write a Python function that transforms a given matrix A using the operation $T^{-1}AS$, where T and S are invertible matrices. The function should first validate if the matrices T and S are invertible, and then perform the transformation.",
        'example': """Example:
        input: A = [[1, 2], [3, 4]], T = [[2, 0], [0, 2]], S = [[1, 1], [0, 1]]
        output: [[0.5, 1.0], [1.5, 2.0]]
        reasoning: The matrices T and S are used to transform matrix A by computing $T^{-1}AS$.""",
        'learn': r'''
<h2>Matrix Transformation using \(T^{-1}AS\)</h2>

Transforming a matrix \(A\) using the operation \(T^{-1}AS\) involves several steps. This operation changes the basis of matrix \(A\) using two invertible matrices \(T\) and \(S\).

Given matrices \(A\), \(T\), and \(S\):

1. **Check if \(T\) and \(S\) are invertible** by ensuring their determinants are non-zero.
2. **Compute the inverses** of \(T\) and \(S\), denoted as \(T^{-1}\) and \(S^{-1}\).
3. **Perform the matrix multiplication** to obtain the transformed matrix:
\[
A' = T^{-1}AS
\]

<h3> Example </h3>
If:
\[
A = \begin{pmatrix} 
1 & 2 \\ 
3 & 4 
\end{pmatrix}
\]
\[
T = \begin{pmatrix} 
2 & 0 \\ 
0 & 2 
\end{pmatrix}
\]
\[
S = \begin{pmatrix} 
1 & 1 \\ 
0 & 1 
\end{pmatrix}
\]

First, check that \(T\) and \(S\) are invertible:
- \(\det(T) = 4 \neq 0 \)
- \(\det(S) = 1 \neq 0 \)

Compute the inverses:
\[
T^{-1} = \begin{pmatrix} 
\frac{1}{2} & 0 \\ 
0 & \frac{1}{2} 
\end{pmatrix}
\]

Then, perform the transformation:
\[
A' = T^{-1}AS = \begin{pmatrix} 
\frac{1}{2} & 0 \\ 
0 & \frac{1}{2} 
\end{pmatrix} \begin{pmatrix} 
1 & 2 \\ 
3 & 4 
\end{pmatrix} \begin{pmatrix} 
1 & 1 \\ 
0 & 1 
\end{pmatrix} = \begin{pmatrix} 
0.5 & 1.0 \\ 
1.5 & 2.0 
\end{pmatrix}
\]
''',
        'starter_code': "import numpy as np\n\ndef transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:\n    return transformed_matrix",
        'solution': """import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    # Convert to numpy arrays for easier manipulation
    A = np.array(A)
    T = np.array(T)
    S = np.array(S)
    
    # Check if the matrices T and S are invertible
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        raise ValueError("The matrices T and/or S are not invertible.")
    
    # Compute the inverses of T and S
    T_inv = np.linalg.inv(T)
    
    # Perform the matrix transformation
    transformed_matrix = T_inv.dot(A).dot(S)
    
    return transformed_matrix.tolist()""",
        'test_cases': [
            {"test": "print(transform_matrix([[1, 2], [3, 4]], [[2, 0], [0, 2]], [[1, 1], [0, 1]]))", "expected_output": "[[0.5, 1.0], [1.5, 2.0]]"},
            {"test": "print(transform_matrix([[1, 0], [0, 1]], [[1, 2], [3, 4]], [[2, 0], [0, 2]]))", "expected_output": "[[0.5, 1.0], [1.5, 2.0]]"},
            {"test": "print(transform_matrix([[2, 3], [1, 4]], [[3, 0], [0, 3]], [[1, 1], [0, 1]]))", "expected_output": "[[0.6666666666666666, 1.0], [0.3333333333333333, 1.3333333333333333]]"},
        ],
    },
    8: {
        'id': 8,
        'title': 'Calculate 2x2 Matrix Inverse (medium)',
        'description': "Write a Python function that calculates the inverse of a 2x2 matrix. Return 'None' if the matrix is not invertible.",
        'example': """Example:
        input: matrix = [[4, 7], [2, 6]]
        output: [[0.6, -0.7], [-0.2, 0.4]]
        reasoning: The inverse of a 2x2 matrix [a, b], [c, d] is given by (1/(ad-bc)) * [d, -b], [-c, a], provided ad-bc is not zero.""",
        'video': "",
        'learn': r'''
<h2>Calculating the Inverse of a 2x2 Matrix</h2>

The inverse of a matrix \(A\) is another matrix, often denoted \(A^{-1}\), such that:
\[
AA^{-1} = A^{-1}A = I
\]
where \(I\) is the identity matrix. For a 2x2 matrix:
\[
A = \begin{pmatrix} 
a & b \\ 
c & d 
\end{pmatrix}
\]

The inverse is:
\[
A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} 
d & -b \\ 
-c & a 
\end{pmatrix}
\]

provided that the determinant \(\det(A) = ad - bc\) is non-zero. If \(\det(A) = 0\), the matrix does not have an inverse.

This process is critical in many applications including solving systems of linear equations, where the inverse is used to find solutions efficiently.
''',
        'starter_code': "def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:\n    return inverse",
        'solution': """def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    determinant = a * d - b * c
    if determinant == 0:
        return None
    inverse = [[d/determinant, -b/determinant], [-c/determinant, a/determinant]]
    return inverse""",
        'test_cases': [
            {"test": "print(inverse_2x2([[4, 7], [2, 6]]))", "expected_output": "[[0.6, -0.7], [-0.2, 0.4]]"},
            {"test": "print(inverse_2x2([[1, 2], [2, 4]]))", "expected_output": "None"},
            {"test": "print(inverse_2x2([[2, 1], [6, 2]]))", "expected_output": "[[-1.0, 0.5], [3.0, -1.0]]"}
        ]
    },
    9: {
        'id': 9,
        'title': 'Matrix times Matrix (medium)',
        'description': "multiply two matrices together (return -1 if shapes of matrix dont aline), i.e. C = A dot product B",
        'example': """ 
Example:
        input: A = [[1,2],
                    [2,4]], 
               B = [[2,1],
                    [3,4]]
        output:[[ 8,  9],
                [16, 18]]
        reasoning: 1*2 + 2*3 = 8;
                   2*2 + 3*4 = 16;
                   1*1 + 2*4 = 9;
                   2*1 + 4*4 = 18
                    
Example 2:
        input: A = [[1,2],
                    [2,4]], 
               B = [[2,1],
                    [3,4],
                    [4,5]]
        output: -1
        reasoning: the length of the rows of A does not equal
          the column length of B""",
        'video':'https://youtu.be/N2j0fA2E9k4',
        'learn': r'''
<h2>Matrix Multiplication</h2>

Consider two matrices \(A\) and \(B\), to demonstrate their multiplication, defined as follows:

- Matrix \(A\):
\[
A = \begin{pmatrix} 
a_{11} & a_{12} \\ 
a_{21} & a_{22} 
\end{pmatrix}
\]

- Matrix \(B\):
\[
B = \begin{pmatrix} 
b_{11} & b_{12} \\ 
b_{21} & b_{22} 
\end{pmatrix}
\]

The multiplication of matrix \(A\) by matrix \(B\) is calculated as:
\[
A \times B = \begin{pmatrix} 
a_{11}b_{11} + a_{12}b_{21} & a_{11}b_{12} + a_{12}b_{22} \\ 
a_{21}b_{11} + a_{22}b_{21} & a_{21}b_{12} + a_{22}b_{22} 
\end{pmatrix}
\]

This operation results in a new matrix where each element is the result of the dot product between the rows of matrix \(A\) and the columns of matrix \(B\).
''',
        'starter_code': """def matrixmul(a:list[list[int|float]],\n              b:list[list[int|float]])-> list[list[int|float]]: \n return c""",
        'solution': """
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    if len(a[0]) != len(b):
        return -1
    
    vals = []
    for i in range(len(a)):
        hold = []
        for j in range(len(b[0])):
            val = 0
            for k in range(len(b)):
                val += a[i][k] * b[k][j]
                           
            hold.append(val)
        vals.append(hold)

    return vals""",
        'test_cases': [
            {"test": "print(matrixmul([[1,2,3],[2,3,4],[5,6,7]],[[3,2,1],[4,3,2],[5,4,3]]))", "expected_output": "[[26, 20, 14], [38, 29, 20], [74, 56, 38]]"},
            {"test": "print(matrixmul([[0,0],[2,4],[1,2]],[[0,0],[2,4]]))", "expected_output": "[[0, 0], [8, 16], [4, 8]]"},
            {"test": "print(matrixmul([[0,0],[2,4],[1,2]],[[0,0,1],[2,4,1],[1,2,3]]))", "expected_output": "-1"},
        ],
    },
    10: {
        'id': 10,
        'title': 'Calculate Covariance Matrix (medium)',
        'description': "Write a Python function that calculates the covariance matrix from a list of vectors. Assume that the input list represents a dataset where each vector is a feature, and vectors are of equal length.",
        'example': """Example:
        input: vectors = [[1, 2, 3], [4, 5, 6]]
        output: [[1.0, 1.0], [1.0, 1.0]]
        reasoning: The dataset has two features with three observations each. The covariance between each pair of features (including covariance with itself) is calculated and returned as a 2x2 matrix.""",
        'video':"https://youtu.be/-BHegXJZAww",
        'learn': r'''
<h2>Calculate Covariance Matrix</h2>

The covariance matrix is a fundamental concept in statistics, illustrating how much two random variables change together. It's essential for understanding the relationships between variables in a dataset.

For a dataset with \(n\) features, the covariance matrix is an \(n \times n\) square matrix where each element (i, j) represents the covariance between the \(i^{th}\) and \(j^{th}\) features. Covariance is defined by the formula:
\[
\text{cov}(X, Y) = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{n-1}
\]

Where:

- \(X\) and \(Y\) are two random variables (features),
- \(x_i\) and \(y_i\) are individual observations of \(X\) and \(Y\),
- \(\bar{x}\) (x-bar) and \(\bar{y}\) (y-bar) are the means of \(X\) and \(Y\),
- \(n\) is the number of observations.

In the covariance matrix:

- The diagonal elements (where \(i = j\)) indicate the variance of each feature.
- The off-diagonal elements show the covariance between different features. This matrix is symmetric, as the covariance between \(X\) and \(Y\) is equal to the covariance between \(Y\) and \(X\), denoted as \(\text{cov}(X, Y) = \text{cov}(Y, X)\).
''',
        'starter_code': "def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:\n    return covariance_matrix",
        'solution': """def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n_features = len(vectors)
    n_observations = len(vectors[0])
    covariance_matrix = [[0 for _ in range(n_features)] for _ in range(n_features)]

    means = [sum(feature) / n_observations for feature in vectors]

    for i in range(n_features):
        for j in range(i, n_features):
            covariance = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) for k in range(n_observations)) / (n_observations - 1)
            covariance_matrix[i][j] = covariance_matrix[j][i] = covariance

    return covariance_matrix""",
        'test_cases': [
            {
                "test": "print(calculate_covariance_matrix([[1, 2, 3], [4, 5, 6]]))",
                "expected_output": "[[1.0, 1.0], [1.0, 1.0]]"
            },
            {
                "test": "print(calculate_covariance_matrix([[1, 5, 6], [2, 3, 4], [7, 8, 9]]))",
                "expected_output": "[[7.0, 2.5, 2.5], [2.5, 1.0, 1.0], [2.5, 1.0, 1.0]]"
            }
        ]
    },
    11: {
        'id': 11,
        'title': 'Solve Linear Equations using Jacobi Method (medium)',
        'description': "Write a Python function that uses the Jacobi method to solve a system of linear equations given by Ax = b. The function should iterate 10 times, rounding each intermediate solution to four decimal places, and return the approximate solution x.",
        'example': """Example:
        input: A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]], b = [-1, 2, 3], n=2
        output: [0.146, 0.2032, -0.5175]
        reasoning: The Jacobi method iteratively solves each equation for x[i] using the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i)), where a_ii is the diagonal element of A and a_ij are the off-diagonal elements.""",
        'video': "https://youtu.be/Y7WSn7K092g",
        'learn': r'''
<h2>Solving Linear Equations Using the Jacobi Method</h2>

The Jacobi method is an iterative algorithm used for solving a system of linear equations \(Ax = b\). This method is particularly useful for large systems where direct methods like Gaussian elimination are computationally expensive.

<h3> Algorithm Overview </h3>

For a system of equations represented by \(Ax = b\), where \(A\) is a matrix and \(x\) and \(b\) are vectors, the Jacobi method involves the following steps:
<ol>
<li><b>Initialization</b>: Start with an initial guess for \(x\).</li>
<li><b>Iteration</b>: For each equation \(i\), update \(x[i]\) using: 
   \[
   x[i] = \frac{1}{a_{ii}} (b[i] - \sum_{j \neq i} a_{ij} x[j])
   \]
   where \(a_{ii}\) are the diagonal elements of \(A\), and \(a_{ij}\) are the off-diagonal elements.</li>
<li><b>Convergence</b>: Repeat the iteration until the changes in \(x\) are below a certain tolerance or until a maximum number of iterations is reached. </li>
</ol>
This method assumes that all diagonal elements of \(A\) are non-zero and that the matrix is diagonally dominant or properly conditioned for convergence.

<h3> Practical Considerations</h3>

- The method may not converge for all matrices.
- Choosing a good initial guess can improve convergence.
- Diagonal dominance of \(A\) ensures convergence of the Jacobi method.
''',
        'starter_code': "import numpy as np\ndef solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:\n    return x",
        'solution': """import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    d_a = np.diag(A)
    nda = A - np.diag(d_a)
    x = np.zeros(len(b))
    x_hold = np.zeros(len(b))
    for _ in range(n):
        for i in range(len(A)):
            x_hold[i] = (1/d_a[i]) * (b[i] - sum(nda[i]*x))
        x = x_hold.copy()
    return np.round(x,4).tolist()""",
        'test_cases': [
            {"test": "print(solve_jacobi(np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]]), np.array([-1, 2, 3]),2))", "expected_output": "[0.146, 0.2032, -0.5175]"},
            {"test": "print(solve_jacobi(np.array([[4, 1, 2], [1, 5, 1], [2, 1, 3]]), np.array([4, 6, 7]),5))", "expected_output": "[-0.0806, 0.9324, 2.4422]"},
            {"test": "print(solve_jacobi(np.array([[4,2,-2],[1,-3,-1],[3,-1,4]]), np.array([0,7,5]),3))", "expected_output": "[1.7083, -1.9583, -0.7812]"}
        ]
    },
    12: {
        'id': 12,
        'title': 'Singular Value Decomposition (SVD) (hard)',
        'description': "Write a Python function that approximates the Singular Value Decomposition on a 2x2 matrix by using the jacobian method and without using numpy svd function, i mean you could but you wouldn't learn anything. return the result in this format.",
        'example': """Example:
        input: a = [[2, 1], [1, 2]]
        output: (array([[-0.70710678, -0.70710678],
                        [-0.70710678,  0.70710678]]),
        array([3., 1.]),
        array([[-0.70710678, -0.70710678],
               [-0.70710678,  0.70710678]]))
        reasoning: U is the first matrix sigma is the second vector and V is the third matrix""",
        'learn': r'''
<h2>Singular Value Decomposition (SVD) via the Jacobi Method</h2>

<p>Singular Value Decomposition (SVD) is a powerful matrix decomposition technique in linear algebra that expresses a matrix as the product of three other matrices, revealing its intrinsic geometric and algebraic properties. When using the Jacobi method, SVD decomposes a matrix \(A\) into:</p>
                        
\[
A = U\Sigma V^T
\]

<ul>
    <li>\(A\) is the original \(m \times n\) matrix.</li>
    <li>\(U\) is an \(m \times m\) orthogonal matrix whose columns are the left singular vectors of \(A\).</li>
    <li>\(\Sigma\) is an \(m \times n\) diagonal matrix containing the singular values of \(A\).</li>
    <li>\(V^T\) is the transpose of an \(n \times n\) orthogonal matrix whose columns are the right singular vectors of \(A\).</li>
</ul>
                        
<h3>The Jacobi Method for SVD</h3>
<p>The Jacobi method is an iterative algorithm used for diagonalizing a symmetric matrix through a series of rotational transformations. It is particularly suited for computing the SVD by iteratively applying rotations to minimize off-diagonal elements until the matrix is diagonal.</p>

<h4>Steps of the Jacobi SVD Algorithm</h4>
<ol>
    <li><strong>Initialization</strong>: Start with \(A^TA\) (or \(AA^T\) for \(U\)) and set \(V\) (or \(U\)) as an identity matrix. The goal is to diagonalize \(A^TA\), obtaining \(V\) in the process.</li>
    <li><strong>Choosing Rotation Targets</strong>: Identify off-diagonal elements in \(A^TA\) to be minimized or zeroed out through rotations.</li>
    <li><strong>Calculating Rotation Angles</strong>: For each target off-diagonal element, calculate the angle \(\theta\) for the Jacobi rotation matrix \(J\) that would zero it. This involves solving for \(\theta\) using \(\text{atan2}\) to accurately handle the quadrant of rotation:
    \[
    \theta = 0.5 \cdot \text{atan2}(2a_{ij}, a_{ii} - a_{jj})
    \]
    where \(a_{ij}\) is the target off-diagonal element, and \(a_{ii}\), \(a_{jj}\) are the diagonal elements of \(A^TA\).
    </li>
    <li><strong>Applying Rotations</strong>: Construct \(J\) using \(\theta\) and apply the rotation to \(A^TA\), effectively reducing the magnitude of the target off-diagonal element. Update \(V\) (or \(U\)) by multiplying it by \(J\).</li>
    <li><strong>Iteration and Convergence</strong>: Repeat the process of selecting off-diagonal elements, calculating rotation angles, and applying rotations until \(A^TA\) is sufficiently diagonalized.</li>
    <li><strong>Extracting SVD Components</strong>: Once diagonalized, the diagonal entries of \(A^TA\) represent the squared singular values of \(A\). The matrices \(U\) and \(V\) are constructed from the accumulated rotations, containing the left and right singular vectors of \(A\), respectively.</li>
</ol>

<h3>Practical Considerations</h3>
<ul>
    <li>The Jacobi method is particularly effective for dense matrices where off-diagonal elements are significant.</li>
    <li>Careful implementation is required to ensure numerical stability and efficiency, especially for large matrices.</li>
    <li>The iterative nature of the Jacobi method makes it computationally intensive, but it is highly parallelizable.</li>
''',
        'starter_code': "import numpy as np \n def svd_2x2_singular_values(A: np.ndarray) -> tuple: \n    return SVD",
        'solution': """import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A_T_A = A.T @ A
    theta = 0.5 * np.arctan2(2 * A_T_A[0, 1], A_T_A[0, 0] - A_T_A[1, 1])
    j = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    A_prime = j.T @ A_T_A @ j 
    
    # Calculate singular values from the diagonalized A^TA (approximation for 2x2 case)
    singular_values = np.sqrt(np.diag(A_prime))

    # Process for AA^T, if needed, similar to A^TA can be added here for completeness
    
    return j, singular_values, j.T""",
        'test_cases': [
            {
                "test": "print(svd_2x2_singular_values(np.array([[2, 1], [1, 2]])))",
                "expected_output": """(array([[ 0.70710678, -0.70710678],
       [ 0.70710678,  0.70710678]]), array([3., 1.]), array([[ 0.70710678,  0.70710678],
       [-0.70710678,  0.70710678]]))"""
            },
            {
                "test": "print(svd_2x2_singular_values(np.array([[1, 2], [3, 4]])))",
                "expected_output": """(array([[ 0.57604844, -0.81741556],
       [ 0.81741556,  0.57604844]]), array([5.4649857 , 0.36596619]), array([[ 0.57604844,  0.81741556],
       [-0.81741556,  0.57604844]]))"""
            }
        ]
    },
    13: {
        'id': 13,
        'title': 'Determinant of a 4x4 Matrix using Laplace\'s Expansion (hard)',
        'description': "Write a Python function that calculates the determinant of a 4x4 matrix using Laplace's Expansion method. The function should take a single argument, a 4x4 matrix represented as a list of lists, and return the determinant of the matrix. The elements of the matrix can be integers or floating-point numbers. Implement the function recursively to handle the computation of determinants for the 3x3 minor matrices.",
        'example': """Example:
        input: a = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        output: 0
        reasoning: Using Laplace's Expansion, the determinant of a 4x4 matrix is calculated by expanding it into minors and cofactors along any row or column. Given the symmetrical and linear nature of this specific matrix, its determinant is 0. The calculation for a generic 4x4 matrix involves more complex steps, breaking it down into the determinants of 3x3 matrices.""",
        'learn': r'''
<h2>Determinant of a 4x4 Matrix using Laplace's Expansion</h2>

Laplace's Expansion, also known as cofactor expansion, is a method to calculate the determinant of a square matrix of any size. For a 4x4 matrix \(A\), this method involves expanding \(A\) into minors and cofactors along a chosen row or column. 

Consider a 4x4 matrix \(A\):
\[
A = \begin{pmatrix}
a_{11} & a_{12} & a_{13} & a_{14} \\
a_{21} & a_{22} & a_{23} & a_{24} \\
a_{31} & a_{32} & a_{33} & a_{34} \\
a_{41} & a_{42} & a_{43} & a_{44}
\end{pmatrix}
\]

The determinant of \(A\), \(\det(A)\), can be calculated by selecting any row or column (e.g., the first row) and using the formula that involves the elements of that row (or column), their corresponding cofactors, and the determinants of the 3x3 minor matrices obtained by removing the row and column of each element. This process is recursive, as calculating the determinants of the 3x3 matrices involves further expansions.

The expansion formula for the first row is as follows:
\[
\det(A) = a_{11}C_{11} - a_{12}C_{12} + a_{13}C_{13} - a_{14}C_{14}
\]

Here, \(C_{ij}\) represents the cofactor of element \(a_{ij}\), which is calculated as \((-1)^{i+j}\) times the determinant of the minor matrix obtained after removing the \(i\)th row and \(j\)th column from \(A\).
''',
        'starter_code': "def determinant_4x4(matrix: list[list[int|float]]) -> float:\n    # Your recursive implementation here\n    pass",
        'solution': """def determinant_4x4(matrix: list[list[int|float]]) -> float:
    # Base case: If the matrix is 1x1, return its single element
    if len(matrix) == 1:
        return matrix[0][0]
    # Recursive case: Calculate determinant using Laplace's Expansion
    det = 0
    for c in range(len(matrix)):
        minor = [row[:c] + row[c+1:] for row in matrix[1:]]  # Remove column c
        cofactor = ((-1)**c) * determinant_4x4(minor)  # Compute cofactor
        det += matrix[0][c] * cofactor  # Add to running total
    return det""",
        'test_cases': [
            {"test": 'print(determinant_4x4([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))', "expected_output": '0'},
            {"test": 'print(determinant_4x4([[4, 3, 2, 1], [3, 2, 1, 4], [2, 1, 4, 3], [1, 4, 3, 2]]))', "expected_output": '-160'},
            {"test": 'print(determinant_4x4([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]))', "expected_output": '0'},
        ]
    },
    14: {
        'id': 14,
        'title': 'Linear Regression Using Normal Equation (easy)',
        'description': "Write a Python function that performs linear regression using the normal equation. The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.",
        'example': """Example:
        input: X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]
        output: [0.0, 1.0]
        reasoning: The linear model is y = 0.0 + 1.0*x, perfectly fitting the input data.""",
        'learn': r'''
<h2>Linear Regression Using the Normal Equation</h2>

Linear regression aims to model the relationship between a scalar dependent variable \(y\) and one or more explanatory variables (or independent variables) \(X\). The normal equation provides an analytical solution to finding the coefficients \(\theta\) that minimize the cost function for linear regression.

Given a matrix \(X\) (with each row representing a training example and each column a feature) and a vector \(y\) (representing the target values), the normal equation is:
\[
\theta = (X^TX)^{-1}X^Ty
\]

Where:
<ul>
<li> \(X^T\) is the transpose of \(X\), </li>
<li> \((X^TX)^{-1}\) is the inverse of the matrix \(X^TX\), </li>
<li> \(y\) is the vector of target values. </li>
</ul>

**Things to note**: This method does not require any feature scaling, and there's no need to choose a learning rate. However, computing the inverse of \(X^TX\) can be computationally expensive if the number of features is very large.

<h3> Practical Implementation</h3>

A practical implementation involves augmenting \(X\) with a column of ones to account for the intercept term and then applying the normal equation directly to compute \(\theta\).
''',
        'starter_code': """import numpy as np\n def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Your code here, make sure to round
    return theta""",
        'solution': """
import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    X_transpose = X.T
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
    theta = np.round(theta, 4).flatten().tolist()
    return theta""",
        'test_cases': [
            {"test": 'print(linear_regression_normal_equation([[1, 1], [1, 2], [1, 3]], [1, 2, 3]))', "expected_output": '[-0.0, 1.0]'},
            {"test": 'print(linear_regression_normal_equation([[1, 3, 4], [1, 2, 5], [1, 3, 2]], [1, 2, 1]))', "expected_output": '[4.0, -1.0, -0.0]'}
        ]
    },
    15: {
        'id': 15,
        'title': 'Linear Regression Using Gradient Descent (easy)',
        'description': "Write a Python function that performs linear regression using gradient descent. The function should take NumPy arrays X (features with a column of ones for the intercept) and y (target) as input, along with learning rate alpha and the number of iterations, and return the coefficients of the linear regression model as a NumPy array. Round your answer to four decimal places. -0.0 is a valid result for rounding a very small number.",
        'example': """Example:
        input: X = np.array([[1, 1], [1, 2], [1, 3]]), y = np.array([1, 2, 3]), alpha = 0.01, iterations = 1000
        output: np.array([0.1107, 0.9513])
        reasoning: The linear model is y = 0.0 + 1.0*x, which fits the input data after gradient descent optimization.""",
        'learn': r'''
<h2>Linear Regression Using Gradient Descent</h2>

Linear regression can also be performed using a technique called gradient descent, where the coefficients (or weights) of the model are iteratively adjusted to minimize a cost function (usually mean squared error). This method is particularly useful when the number of features is too large for analytical solutions like the normal equation or when the feature matrix is not invertible.

The gradient descent algorithm updates the weights by moving in the direction of the negative gradient of the cost function with respect to the weights. The updates occur iteratively until the algorithm converges to a minimum of the cost function.

The update rule for each weight is given by:
\[
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)x_j^{(i)}
\]
Where:
<ul>
<li> \(\alpha\) is the learning rate, </li>
<li> \(m\) is the number of training examples,</li>
<li> \(h_{\theta}(x^{(i)})\) is the hypothesis function at iteration \(i\),</li>
<li> \(x^{(i)}\) is the feature vector of the \(i^{th}\) training example,</li>
<li> \(y^{(i)}\) is the actual target value for the \(i^{th}\) training example,</li>
<li> \(x_j^{(i)}\) is the value of feature \(j\) for the \(i^{th}\) training example.</li>
</ul>
**Things to note**: The choice of learning rate and the number of iterations are crucial for the convergence and performance of gradient descent. Too small a learning rate may lead to slow convergence, while too large a learning rate may cause overshooting and divergence.

<h3> Practical Implementation </h3>

Implementing gradient descent involves initializing the weights, computing the gradient of the cost function, and iteratively updating the weights according to the update rule.
''',
        'starter_code': """import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    # Your code here, make sure to round
    m, n = X.shape
    theta = np.zeros((n, 1))
    return theta""",
        'solution': """
import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape
    theta = np.zeros((n, 1))
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y.reshape(-1, 1)
        updates = X.T @ errors / m
        theta -= alpha * updates
    return np.round(theta.flatten(), 4)""",
        'test_cases': [
            {
                "test": "print(linear_regression_gradient_descent(np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000))",
                "expected_output": "[0.1107 0.9513]"
            },
            {
                "test": "print(linear_regression_gradient_descent(np.array([[1, 1, 3], [1, 2, 4], [1, 3, 5]]), np.array([2, 3, 5]), 0.1, 10))",
                "expected_output": "[-1.0241 -1.9133 -3.9616]"
            }
        ]
    },
    16: {
        'id': 16,
        'title': 'Feature Scaling Implementation (easy)',
        'description': "Write a Python function that performs feature scaling on a dataset using both standardization and min-max normalization. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. It should return two 2D NumPy arrays: one scaled by standardization and one by min-max normalization. Make sure all results are rounded to the nearest 4th decimal.",
        'example': """Example:
        input: data = np.array([[1, 2], [3, 4], [5, 6]])
        output: ([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        reasoning: Standardization rescales the feature to have a mean of 0 and a standard deviation of 1. Min-max normalization rescales the feature to a range of [0, 1], where the minimum feature value maps to 0 and the maximum to 1.""",
        'learn': r'''
<h2>Feature Scaling Techniques</h2>

Feature scaling is crucial in many machine learning algorithms that are sensitive to the magnitude of features. This includes algorithms that use distance measures like k-nearest neighbors and gradient descent-based algorithms like linear regression.

<h3> Standardization: </h3>
Standardization (or Z-score normalization) is the process where the features are rescaled so that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one:
\[
z = \frac{(x - \mu)}{\sigma}
\]
Where \(x\) is the original feature, \(\mu\) is the mean of that feature, and \(\sigma\) is the standard deviation.

<h3> Min-Max Normalization: </h3>
Min-max normalization rescales the feature to a fixed range, typically 0 to 1, or it can be shifted to any range \([a, b]\) by transforming the data according to the formula:
\[
x' = \frac{(x - \text{min}(x))}{(\text{max}(x) - \text{min}(x))} \times (\text{max} - \text{min}) + \text{min}
\]
Where \(x\) is the original value, \(\text{min}(x)\) is the minimum value for that feature, \(\text{max}(x)\) is the maximum value, and \(\text{min}\) and \(\text{max}\) are the new minimum and maximum values for the scaled data.

Implementing these scaling techniques will ensure that the features contribute equally to the development of the model and improve the convergence speed of learning algorithms.
''',
        'starter_code': """def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    return standardized_data, normalized_data""",
        'solution': """
import numpy as np

def feature_scaling(data):
    # Standardization
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    
    # Min-Max Normalization
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    
    return np.round(standardized_data,4).tolist(), np.round(normalized_data,4).tolist()""",
        'test_cases': [
            {
                "test": "print(feature_scaling(np.array([[1, 2], [3, 4], [5, 6]])))",
                "expected_output": "([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])"
            }
        ]
    },
    17: {
        'id': 17,
        'title': 'K-Means Clustering (medium)',
        'description': "Write a Python function that implements the k-Means algorithm for clustering, starting with specified initial centroids and a set number of iterations. The function should take a list of points (each represented as a tuple of coordinates), an integer k representing the number of clusters to form, a list of initial centroids (each a tuple of coordinates), and an integer representing the maximum number of iterations to perform. The function will iteratively assign each point to the nearest centroid and update the centroids based on the assignments until the centroids do not change significantly, or the maximum number of iterations is reached. The function should return a list of the final centroids of the clusters. Round to the nearest fourth decimal.",
        'example': """Example:
        input: points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], k = 2, initial_centroids = [(1, 1), (10, 1)], max_iterations = 10
        output: [(1, 2), (10, 2)]
        reasoning: Given the initial centroids and a maximum of 10 iterations,
        the points are clustered around these points, and the centroids are
        updated to the mean of the assigned points, resulting in the final
        centroids which approximate the means of the two clusters.
        The exact number of iterations needed may vary,
        but the process will stop after 10 iterations at most.""",
        'learn': r'''
<h2>Implementing k-Means Clustering</h2>

k-Means clustering is a method to partition `n` points into `k` clusters. Here is a brief overview of how to implement the k-Means algorithm:

1. **Initialization**: Start by selecting `k` initial centroids. These can be randomly selected from the dataset or based on prior knowledge.

2. **Assignment Step**: For each point in the dataset, find the nearest centroid. The "nearest" can be defined using Euclidean distance. Assign the point to the cluster represented by this nearest centroid.

3. **Update Step**: Once all points are assigned to clusters, update the centroids by calculating the mean of all points in each cluster. This becomes the new centroid of the cluster.

4. **Iteration**: Repeat the assignment and update steps until the centroids no longer change significantly, or until a predetermined number of iterations have been completed. This iterative process helps in refining the clusters to minimize within-cluster variance.

5. **Result**: The final centroids represent the center of the clusters, and the points are partitioned accordingly.

This algorithm assumes that the `mean` is a meaningful measure, which might not be the case for non-numeric data. The choice of initial centroids can significantly affect the final clusters, hence multiple runs with different starting points can lead to a more comprehensive understanding of the cluster structure in the data.
''',
        'starter_code': """def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    # Your code here
    return final_centroids""",
        'solution': """
import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))

def k_means_clustering(points, k, initial_centroids, max_iterations):
    points = np.array(points)
    centroids = np.array(initial_centroids)
    
    for iteration in range(max_iterations):
        # Assign points to the nearest centroid
        distances = np.array([euclidean_distance(points, centroid) for centroid in centroids])
        assignments = np.argmin(distances, axis=0)

        new_centroids = np.array([points[assignments == i].mean(axis=0) if len(points[assignments == i]) > 0 else centroids[i] for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        centroids = np.round(centroids,4)
    return [tuple(centroid) for centroid in centroids]""",
        'test_cases': [
            {
                "test": "print(k_means_clustering([(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], 2, [(1, 1), (10, 1)], 10))",
                "expected_output": "[(1.0, 2.0), (10.0, 2.0)]"
            },
            {
                "test": "print(k_means_clustering([(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)], 2, [(1, 1, 1), (10, 10, 10)], 10))",
                "expected_output": "[(1.0, 1.0, 1.0), (10.3333, 10.6667, 10.3333)]"
            }
        ]
    },
    18: {
        'id': 18,
        'title': 'Cross-Validation Data Split Implementation (medium)',
        'description': "Write a Python function that performs k-fold cross-validation data splitting from scratch. The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature) and an integer k representing the number of folds. The function should split the dataset into k parts, systematically use one part as the test set and the remaining as the training set, and return a list where each element is a tuple containing the training set and test set for each fold.",
        'example': """Example:
        input: data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), k = 5
        output: [[[[3, 4], [5, 6], [7, 8], [9, 10]], [[1, 2]]],
                [[[1, 2], [5, 6], [7, 8], [9, 10]], [[3, 4]]],
                [[[1, 2], [3, 4], [7, 8], [9, 10]], [[5, 6]]], 
                [[[1, 2], [3, 4], [5, 6], [9, 10]], [[7, 8]]], 
                [[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10]]]]
        reasoning: The dataset is divided into 5 parts, each being used once as a test set while the remaining parts serve as the training set.""",
        'learn': r'''
<h2>Understanding k-Fold Cross-Validation Data Splitting</h2>

k-Fold cross-validation is a technique used to evaluate the generalizability of a model by dividing the data into `k` folds or subsets. Each fold acts as a test set once, with the remaining `k-1` folds serving as the training set. This approach ensures that every data point gets used for both training and testing, improving model validation.

<h3> Steps in k-Fold Cross-Validation Data Split: </h3>

    <ol>
        <li><strong>Shuffle the dataset randomly</strong>. (but not in this case because we test for a unique result)</li>
        <li><strong>Split the dataset into k groups</strong>.</li>
        <li><strong>Generate Data Splits</strong>: For each group, treat that group as the test set and the remaining groups as the training set.</li>
    </ol>

<h3> Benefits of this Approach: </h3>

- Ensures all data is used for both training and testing.
- Reduces bias since each data point gets to be in a test set exactly once.
- Provides a more robust estimate of model performance.

Implementing this data split function will allow a deeper understanding of how data partitioning affects machine learning models and will provide a foundation for more complex validation techniques.
''',
        'starter_code': """def cross_validation_split(data: np.ndarray, k: int) -> list:
    # Your code here
    return folds""",
        'solution': """
import numpy as np

def cross_validation_split(data, k):
    np.random.shuffle(data)  # This line can be removed if shuffling is not desired in examples
    fold_size = len(data) // k
    folds = []
    
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size if i != k-1 else len(data)
        test = data[start:end]
        train = np.concatenate([data[:start], data[end:]])
        folds.append([train.tolist(), test.tolist()])
    
    return folds""",
        'test_cases': [
            {
                "test": "print(cross_validation_split(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), 2))",
                "expected_output": 
                    """[[[[5, 6], [7, 8], [9, 10]], [[1, 2], [3, 4]]], [[[1, 2], [3, 4]], [[5, 6], [7, 8], [9, 10]]]]"""
            }
        ]
    },
    19: {
        'id': 19,
        'title': 'Principal Component Analysis (PCA) Implementation (medium)',
        'description': "Write a Python function that performs Principal Component Analysis (PCA) from scratch. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. The function should standardize the dataset, compute the covariance matrix, find the eigenvalues and eigenvectors, and return the principal components (the eigenvectors corresponding to the largest eigenvalues). The function should also take an integer k as input, representing the number of principal components to return.",
        'example': """Example:
        input: data = np.array([[1, 2], [3, 4], [5, 6]]), k = 1
        output:  [[0.7071], [0.7071]]
        reasoning: After standardizing the data and computing the covariance matrix, the eigenvalues and eigenvectors are calculated. The largest eigenvalue's corresponding eigenvector is returned as the principal component, rounded to four decimal places.""",
        'learn': r'''
<h2>Understanding Eigenvalues in PCA</h2>

Principal Component Analysis (PCA) utilizes the concept of eigenvalues and eigenvectors to identify the principal components of a dataset. Here's how eigenvalues fit into the PCA process:

<h3> Eigenvalues and Eigenvectors: The Foundation of PCA </h3>

For a given square matrix \(A\), representing the covariance matrix in PCA, eigenvalues \(\lambda\) and their corresponding eigenvectors \(v\) satisfy:
\[
Av = \lambda v
\]

<h3> Calculating Eigenvalues </h3>

The eigenvalues of matrix \(A\) are found by solving the characteristic equation:
\[
\det(A - \lambda I) = 0
\]
where \(I\) is the identity matrix of the same dimension as \(A\). This equation highlights the relationship between a matrix, its eigenvalues, and eigenvectors.

<h3> Role in PCA </h3>

In PCA, the covariance matrix's eigenvalues represent the variance explained by its eigenvectors. Thus, selecting the eigenvectors associated with the largest eigenvalues is akin to choosing the principal components that retain the most data variance.

<h3> Eigenvalues and Dimensionality Reduction </h3>

The magnitude of an eigenvalue correlates with the importance of its corresponding eigenvector (principal component) in representing the dataset's variability. By selecting a subset of eigenvectors corresponding to the largest eigenvalues, PCA achieves dimensionality reduction while preserving as much of the dataset's variability as possible.

<h3> Practical Application </h3>

    <ol>
        <li><strong>Standardize the Dataset</strong>: Ensure that each feature has a mean of 0 and a standard deviation of 1.</li>
        <li><strong>Compute the Covariance Matrix</strong>: Reflects how features vary together.</li>
        <li><strong>Find Eigenvalues and Eigenvectors</strong>: Solve the characteristic equation for the covariance matrix.</li>
        <li><strong>Select Principal Components</strong>: Choose eigenvectors (components) with the highest eigenvalues for dimensionality reduction.</li>
    </ol>
Through this process, PCA transforms the original features into a new set of uncorrelated features (principal components), ordered by the amount of original variance they explain.
''',
        'starter_code': """import numpy as np \ndef pca(data: np.ndarray, k: int) -> list[list[int|float]]:
    # Your code here
    return principal_components""",
        'solution': """
import numpy as np

def pca(data, k):
    # Standardize the data
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(data_standardized, rowvar=False)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort the eigenvectors by decreasing eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:,idx]
    
    # Select the top k eigenvectors (principal components)
    principal_components = eigenvectors_sorted[:, :k]
    
    return np.round(principal_components, 4).tolist()""",
        'test_cases': [
            {
                "test": "print(pca(np.array([[4,2,1],[5,6,7],[9,12,1],[4,6,7]]),2))",
                "expected_output": "[[0.6855, 0.0776], [0.6202, 0.4586], [-0.3814, 0.8853]]"
            },
            {
                "test": "print(pca(np.array([[1, 2], [3, 4], [5, 6]]), k = 1))",
                "expected_output": " [[0.7071], [0.7071]]"
            }
        ]
    },
    20: {
        'id': 20,
        'title': 'Decision Tree Learning (hard)',
        'description': "Write a Python function that implements the decision tree learning algorithm for classification. The function should use recursive binary splitting based on entropy and information gain to build a decision tree. It should take a list of examples (each example is a dict of attribute-value pairs) and a list of attribute names as input, and return a nested dictionary representing the decision tree.",
        'example': """Example:
        input: examples = [
                    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'No'},
                    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'No'},
                    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
                    {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'}
                ],
                attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
        output: {
            'Outlook': {
                'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}},
                'Overcast': 'Yes',
                'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}
            }
        }
        reasoning: Using the given examples, the decision tree algorithm determines that 'Outlook' is the best attribute to split the data initially. When 'Outlook' is 'Overcast', the outcome is always 'Yes', so it becomes a leaf node. In cases of 'Sunny' and 'Rain', it further splits based on 'Humidity' and 'Wind', respectively. The resulting tree structure is able to classify the training examples with the attributes 'Outlook', 'Temperature', 'Humidity', and 'Wind'.
        """,
        'learn': r'''
<h2>Decision Tree Learning Algorithm</h2>

The decision tree learning algorithm is a method used for classification that predicts the value of a target variable based on several input variables. Each internal node of the tree corresponds to an input variable, and each leaf node corresponds to a class label.

The recursive binary splitting starts by selecting the attribute that best separates the examples according to the entropy and information gain, which are calculated as follows:

Entropy: \(H(X) = -\sum p(x) \log_2 p(x)\) 

Information Gain: \(IG(D, A) = H(D) - \sum \frac{|D_v|}{|D|} H(D_v)\)

Where:
- \(H(X)\) is the entropy of the set,
- \(IG(D, A)\) is the information gain of dataset \(D\) after splitting on attribute \(A\),
- \(D_v\) is the subset of \(D\) for which attribute \(A\) has value \(v\).

The attribute with the highest information gain is used at each step, and the dataset is split based on this attribute's values. This process continues recursively until all data is perfectly classified or no remaining attributes can be used to make a split.
''',
        'starter_code': """def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    # Your code here
    return decision_tree""",
        'solution': """
import math
from collections import Counter

def calculate_entropy(labels):
    label_counts = Counter(labels)
    total_count = len(labels)
    entropy = -sum((count / total_count) * math.log2(count / total_count) for count in label_counts.values())
    return entropy

def calculate_information_gain(examples, attr, target_attr):
    total_entropy = calculate_entropy([example[target_attr] for example in examples])
    values = set(example[attr] for example in examples)
    attr_entropy = 0
    for value in values:
        value_subset = [example[target_attr] for example in examples if example[attr] == value]
        value_entropy = calculate_entropy(value_subset)
        attr_entropy += (len(value_subset) / len(examples)) * value_entropy
    return total_entropy - attr_entropy

def majority_class(examples, target_attr):
    return Counter([example[target_attr] for example in examples]).most_common(1)[0][0]

def learn_decision_tree(examples, attributes, target_attr):
    if not examples:
        return 'No examples'
    if all(example[target_attr] == examples[0][target_attr] for example in examples):
        return examples[0][target_attr]
    if not attributes:
        return majority_class(examples, target_attr)
    
    gains = {attr: calculate_information_gain(examples, attr, target_attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    tree = {best_attr: {}}
    
    for value in set(example[best_attr] for example in examples):
        subset = [example for example in examples if example[best_attr] == value]
        new_attributes = attributes.copy()
        new_attributes.remove(best_attr)
        subtree = learn_decision_tree(subset, new_attributes, target_attr)
        tree[best_attr][value] = subtree
    
    return tree
""",
        'test_cases': [
            {
                "test": "print(learn_decision_tree([\n"
                        "    {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'},\n"
                        "    {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},\n"
                        "    {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},\n"
                        "    {'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},\n"
                        "    {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'Yes'},\n"
                        "    {'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'},\n"
                        "    {'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'},\n"
                        "    {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'}\n"
                        "], ['Outlook', 'Wind'], 'PlayTennis'))",
                "expected_output":  "{'Outlook': {'Sunny': {'Wind': {'Weak': 'No', 'Strong': 'No'}}, 'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}, 'Overcast': 'Yes'}}"
            }
        ]
    },
    21: {
        'id': 21,
        'title': 'Pegasos Kernel SVM Implementation (advanced)',
        'description': "Write a Python function that implements the Pegasos algorithm to train a kernel SVM classifier from scratch. The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature), a label vector (1D NumPy array where each entry corresponds to the label of the sample), and training parameters such as the choice of kernel (linear or RBF), regularization parameter (lambda), and the number of iterations. The function should perform binary classification and return the model's alpha coefficients and bias.",
        'example': """Example:
        input: data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), labels = np.array([1, 1, -1, -1]), kernel = 'rbf', lambda_val = 0.01, iterations = 100
        output: alpha = [0.03, 0.02, 0.05, 0.01], b = -0.05
        reasoning: Using the RBF kernel, the Pegasos algorithm iteratively updates the weights based on a sub-gradient descent method, taking into account the non-linear separability of the data induced by the kernel transformation.""",
        'learn': r'''
<h2>Pegasos Algorithm and Kernel SVM</h2>

The Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) algorithm is a simple and efficient stochastic gradient descent method designed for solving the SVM optimization problem in its primal form.

<h3> Key Concepts: </h3>
<ul>
<li><b>Kernel Trick</b>: Allows SVM to classify data that is not linearly separable by implicitly mapping input features into high-dimensional feature spaces.</li>
<li><b>Regularization Parameter (\(\lambda\))</b>: Controls the trade-off between achieving a low training error and a low model complexity.</li>
<li> <b>Sub-Gradient Descent</b>: Used in the Pegasos algorithm to optimize the objective function, which includes both the hinge loss and a regularization term.</li>
</ul>
<h3> Steps in the Pegasos Algorithm: </h3>

1. **Initialize Parameters**: Start with zero weights and choose an appropriate value for the regularization parameter \( \lambda \).
2. **Iterative Updates**: For each iteration and for each randomly selected example, update the model parameters using the learning rule derived from the sub-gradient of the loss function.
3. **Kernel Adaptation**: Use the chosen kernel to compute the dot products required in the update step, allowing for non-linear decision boundaries.

<h3> Practical Implementation: </h3>

The implementation involves selecting a kernel function, calculating the kernel matrix, and performing iterative updates on the alpha coefficients according to the Pegasos rule:
\[
\alpha_{t+1} = (1 - \eta_t \lambda) \alpha_t + \eta_t (y_i K(x_i, x))
\]
where \( \eta_t \) is the learning rate at iteration \( t \), and \( K \) denotes the kernel function.

This method is particularly well-suited for large-scale learning problems due to its efficient use of data and incremental learning nature.
''',
        'starter_code': """def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100) -> (list, float):
    # Your code here
    return alphas, b""",
        'solution': """
import numpy as np

def linear_kernel(x, y):
    return np.dot(x, y)

def rbf_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def pegasos_kernel_svm(data, labels, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0):
    n_samples = len(data)
    alphas = np.zeros(n_samples)
    b = 0

    for t in range(1, iterations + 1):
        for i in range(n_samples):
            eta = 1.0 / (lambda_val * t)
            if kernel == 'linear':
                kernel_func = linear_kernel
            elif kernel == 'rbf':
                kernel_func = lambda x, y: rbf_kernel(x, y, sigma)
    
            decision = sum(alphas[j] * labels[j] * kernel_func(data[j], data[i]) for j in range(n_samples)) + b
            if labels[i] * decision < 1:
                alphas[i] += eta * (labels[i] - lambda_val * alphas[i])
                b += eta * labels[i]

    return np.round(alphas,4).tolist(), np.round(b,4)""",
        'test_cases': [
            {
                "test": "print(pegasos_kernel_svm(np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), np.array([1, 1, -1, -1]), kernel='linear', lambda_val=0.01, iterations=100))",
                "expected_output": "([100.0, 0.0, -100.0, -100.0], -937.4755)"
            },
            {
                "test": "print(pegasos_kernel_svm(np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), np.array([1, 1, -1, -1]), kernel='rbf', lambda_val=0.01, iterations=100, sigma=0.5))",
                "expected_output": "([100.0, 99.0, -100.0, -100.0], -115.0)"
            }
        ]
    },22: {
        'id': 22,
        'title': 'Sigmoid Activation Function Understanding (easy)',
        'description': "Write a Python function that computes the output of the sigmoid activation function given an input value z. The function should return the output rounded to four decimal places.",
        'example': """Example:
        input: z = 0
        output: 0.5
        reasoning: The sigmoid function is defined as σ(z) = 1 / (1 + exp(-z)). For z = 0, exp(-0) = 1, hence the output is 1 / (1 + 1) = 0.5.""",
        'learn': r'''
<h2>Understanding the Sigmoid Activation Function</h2>

The sigmoid activation function is crucial in neural networks, especially for binary classification tasks. It maps any real-valued number into the (0, 1) interval, making it useful for modeling probability as an output.

<h3>Mathematical Definition</h3>

The sigmoid function is mathematically defined as:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Where \(z\) is the input to the function.

<h3>Characteristics</h3>

<ul>
    <li><strong>Output Range:</strong> The output is always between 0 and 1.</li>
    <li><strong>Shape:</strong> It has an "S" shaped curve.</li>
    <li><strong>Gradient:</strong> The function's gradient is highest near \(z = 0\) and decreases toward either end of the z-axis.</li>
</ul>

This function is particularly useful for turning logits (raw prediction values) into probabilities in binary classification models.
''',
        'starter_code': """import math\n\ndef sigmoid(z: float) -> float:\n    # Your code here\n    return result""",
        'solution': """
import math
def sigmoid(z: float) -> float:
    result = 1 / (1 + math.exp(-z))
    return round(result, 4)""",
        'test_cases': [
            {
                "test": "print(sigmoid(0))",
                "expected_output": "0.5"
            },
            {
                "test": "print(sigmoid(1))",
                "expected_output": "0.7311"
            },
            {
                "test": "print(sigmoid(-1))",
                "expected_output": "0.2689"
            }
        ]
    },
    23: {
        'id': 23,
        'title': 'Softmax Activation Function Implementation (easy)',
        'description': "Write a Python function that computes the softmax activation for a given list of scores. The function should return the softmax values as a list, each rounded to four decimal places.",
        'example': """Example:
        input: scores = [1, 2, 3]
        output: [0.0900, 0.2447, 0.6652]
        reasoning: The softmax function converts a list of values into a probability distribution. The probabilities are proportional to the exponential of each element divided by the sum of the exponentials of all elements in the list.""",
        'learn': r'''
<h2>Understanding the Softmax Activation Function</h2>

The softmax function is a generalization of the sigmoid function and is used in the output layer of a neural network model that handles multi-class classification tasks.

<h3>Mathematical Definition</h3>

The softmax function is mathematically represented as:

\[
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
\]

<h3>Characteristics</h3>

<ul>
    <li><strong>Output Range:</strong> Each output value is between 0 and 1, and the sum of all outputs is 1.</li>
    <li><strong>Purpose:</strong> It transforms scores into probabilities, which are easier to interpret and are useful for classification.</li>
</ul>

This function is essential for models where the output needs to represent a probability distribution across multiple classes.
''',
        'starter_code': """import math\n\ndef softmax(scores: list[float]) -> list[float]:\n    # Your code here\n    return probabilities""",
        'solution': """
import math
def softmax(scores: list[float]) -> list[float]:
    exp_scores = [math.exp(score) for score in scores]
    sum_exp_scores = sum(exp_scores)
    probabilities = [round(score / sum_exp_scores, 4) for score in exp_scores]
    return probabilities""",
        'test_cases': [
            {
                "test": "print(softmax([1, 2, 3]))",
                "expected_output": "[0.09, 0.2447, 0.6652]"
            },
            {
                "test": "print(softmax([1, 1, 1]))",
                "expected_output": "[0.3333, 0.3333, 0.3333]"
            },
            {
                "test": "print(softmax([-1, 0, 5]))",
                "expected_output": "[0.0025, 0.0067, 0.9909]"
            }
        ]
    },
    24: {
        'id': 24,
        'title': 'Single Neuron (easy)',
        'description': "Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, handling multidimensional input features. The function should take a list of feature vectors (each vector representing multiple features for an example), associated true binary labels, and the neuron's weights (one for each feature) and bias as input. It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels, both rounded to four decimal places.",
        'example': """Example:
        input: features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], labels = [0, 1, 0], weights = [0.7, -0.4], bias = -0.1
        output: ([0.4626, 0.4134, 0.6682], 0.3349)
        reasoning: For each input vector, the weighted sum is calculated by multiplying each feature by its corresponding weight, adding these up along with the bias, then applying the sigmoid function to produce a probability. The MSE is calculated as the average squared difference between each predicted probability and the corresponding true label.""",
        'learn': r'''
<h2>Single Neuron Model with Multidimensional Input and Sigmoid Activation</h2>

This task involves a neuron model designed for binary classification with multidimensional input features, using the sigmoid activation function to output probabilities. It also involves calculating the mean squared error (MSE) to evaluate prediction accuracy.

<h3>Mathematical Background</h3>

<ul>
    <li><strong>Neuron Output Calculation</strong>:
        \[
        z = \sum (weight_i \times feature_i) + bias
        \]
        \[
        \sigma(z) = \frac{1}{1 + e^{-z}}
        \]
    </li>

    <li><strong>MSE Calculation</strong>:
        \[
        MSE = \frac{1}{n} \sum (predicted - true)^2
        \]
        Where:
        <ul>
            <li>\(z\) is the sum of weighted inputs plus bias,</li>
            <li>\(\sigma(z)\) is the sigmoid activation output,</li>
            <li>\(predicted\) are the probabilities after sigmoid activation,</li>
            <li>\(true\) are the true binary labels.</li>
        </ul>
    </li>
</ul>

<h3>Practical Implementation</h3>

<ul>
    <li>Each feature vector is processed to calculate a combined weighted sum, which is then passed through the sigmoid function to determine the probability of the input belonging to the positive class.</li>
    <li>MSE provides a measure of error, offering insights into the model's performance and aiding in its optimization.</li>
</ul>
''',
        'starter_code': """import math\n\ndef single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):\n    # Your code here\n    return probabilities, mse""",
        'solution': """
import math
def single_neuron_model(features, labels, weights, bias):
    probabilities = []
    for feature_vector in features:
        z = sum(weight * feature for weight, feature in zip(weights, feature_vector)) + bias
        prob = 1 / (1 + math.exp(-z))
        probabilities.append(round(prob, 4))
    
    mse = sum((prob - label) ** 2 for prob, label in zip(probabilities, labels)) / len(labels)
    mse = round(mse, 4)
    
    return probabilities, mse""",
        'test_cases': [
            {
                "test": "print(single_neuron_model([[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], [0, 1, 0], [0.7, -0.4], -0.1))",
                "expected_output": "([0.4626, 0.4134, 0.6682], 0.3349)"
            },
            {
                "test": "print(single_neuron_model([[1, 2], [2, 3], [3, 1]], [1, 0, 1], [0.5, -0.2], 0))",
                "expected_output": "([0.525, 0.5987, 0.7858], 0.21)"
            }
        ]
    },
    25: {
        'id': 25,
        'title': 'Single Neuron with Backpropagation (medium)',
        'description': "Write a Python function that simulates a single neuron with sigmoid activation, and implements backpropagation to update the neuron's weights and bias. The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs. The function should update the weights and bias using gradient descent based on the MSE loss, and return the updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.",
        'example': """Example:
        input: features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], labels = [1, 0, 0], initial_weights = [0.1, -0.2], initial_bias = 0.0, learning_rate = 0.1, epochs = 2
        output: updated_weights = [0.0808, -0.1916], updated_bias = -0.0214, mse_values = [0.2386, 0.2348]
        reasoning: The neuron receives feature vectors and computes predictions using the sigmoid activation. Based on the predictions and true labels, the gradients of MSE loss with respect to weights and bias are computed and used to update the model parameters across epochs.""",
        'learn': r'''
<h2>Neural Network Learning with Backpropagation</h2>

This question involves implementing backpropagation for a single neuron in a neural network. The neuron processes inputs and updates parameters to minimize the Mean Squared Error (MSE) between predicted outputs and true labels.

<h3>Mathematical Background</h3>

<ul>
    <li><strong>Forward Pass</strong>:
        <ul>
            <li>Compute the neuron output by calculating the dot product of the weights and input features and adding the bias:
                \[
                z = w_1x_1 + w_2x_2 + ... + w_nx_n + b
                \]
                \[
                \sigma(z) = \frac{1}{1 + e^{-z}}
                \]
            </li>
        </ul>
    </li>

    <li><strong>Loss Calculation (MSE)</strong>:
        <ul>
            <li>The Mean Squared Error is used to quantify the error between the neuron's predictions and the actual labels:
                \[
                MSE = \frac{1}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i)^2
                \]
            </li>
        </ul>
    </li>

    <li><strong>Backward Pass (Gradient Calculation)</strong>:
        <ul>
            <li>Compute the gradient of the MSE with respect to each weight and the bias. This involves the partial derivatives of the loss function with respect to the output of the neuron, multiplied by the derivative of the sigmoid function:
                \[
                \frac{\partial MSE}{\partial w_j} = \frac{2}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i) \sigma'(z_i) x_{ij}
                \]
                \[
                \frac{\partial MSE}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i) \sigma'(z_i)
                \]
            </li>
        </ul>
    </li>

    <li><strong>Parameter Update</strong>:
        <ul>
            <li>Update each weight and the bias by subtracting a portion of the gradient determined by the learning rate:
                \[
                w_j = w_j - \alpha \frac{\partial MSE}{\partial w_j}
                \]
                \[
                b = b - \alpha \frac{\partial MSE}{\partial b}
                \]
            </li>
        </ul>
    </li>
</ul>

<h3>Practical Implementation</h3>

This process refines the neuron's ability to predict accurately by iteratively adjusting the weights and bias based on the error gradients, optimizing the neural network's performance over multiple iterations.
''',
        'starter_code': """import numpy as np\n\ndef train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):\n    # Your code here\n    return updated_weights, updated_bias, mse_values""",
        'solution': """
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    weights = np.array(initial_weights)
    bias = initial_bias
    features = np.array(features)
    labels = np.array(labels)
    mse_values = []

    for _ in range(epochs):
        z = np.dot(features, weights) + bias
        predictions = sigmoid(z)
        
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(mse, 4))

        # Gradient calculation for weights and bias
        errors = predictions - labels
        weight_gradients = np.dot(features.T, errors * predictions * (1 - predictions))
        bias_gradient = np.sum(errors * predictions * (1 - predictions))
        
        # Update weights and bias
        weights -= learning_rate * weight_gradients / len(labels)
        bias -= learning_rate * bias_gradient / len(labels)

        # Round weights and bias for output
        updated_weights = np.round(weights, 4)
        updated_bias = round(bias, 4)

    return updated_weights.tolist(), updated_bias, mse_values""",
        'test_cases': [
            {
                "test": "print(train_neuron(np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]), np.array([1, 0, 0]), np.array([0.1, -0.2]), 0.0, 0.1, 2))",
                "expected_output": "([0.1019, -0.1711], -0.0083, [0.3033, 0.2987])"
            },
            {
                "test": "print(train_neuron(np.array([[1, 2], [2, 3], [3, 1]]), np.array([1, 0, 1]), np.array([0.5, -0.2]), 0, 0.1, 3))",
                "expected_output": "([0.4943, -0.2155], 0.0013, [0.21, 0.2093, 0.2087])"
            }
        ]
    },
    26: {
        'id': 26,
        'title': 'Implementing Basic Autograd Operations (medium)',
        'description': "Special thanks to Andrej Karpathy for making a video about this, if you haven't already check out his videos on YouTube https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg. Write a Python class similar to the provided 'Value' class that implements the basic autograd operations: addition, multiplication, and ReLU activation. The class should handle scalar values and should correctly compute gradients for these operations through automatic differentiation.",
        'example': """Example:
        a = Value(2)
        b = Value(-3)
        c = Value(10)
        d = a + b * c
        e = d.relu()
        e.backward()
        print(a, b, c, d, e)
        Output: Value(data=2, grad=0) Value(data=-3, grad=10) Value(data=10, grad=-3) Value(data=-28, grad=1) Value(data=0, grad=1)
        Explanation: The output reflects the forward computation and gradients after backpropagation. The ReLU on 'd' zeros out its output and gradient due to the negative data value.""",
        'learn': r'''
<h2>Understanding Mathematical Concepts in Autograd Operations</h2>

First off watch this: https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg

This task focuses on the implementation of basic automatic differentiation mechanisms for neural networks. The operations of addition, multiplication, and ReLU are fundamental to neural network computations and their training through backpropagation.

<h3>Mathematical Foundations</h3>

<ul>
    <li><strong>Addition (`__add__`)</strong>:
        <ul>
            <li><strong>Forward pass</strong>: For two scalar values \(a\) and \(b\), their sum \(s\) is simply \(s = a + b\).</li>
            <li><strong>Backward pass</strong>: The derivative of \(s\) with respect to both \(a\) and \(b\) is 1. Therefore, during backpropagation, the gradient of the output is passed directly to both inputs.</li>
        </ul>
    </li>

    <li><strong>Multiplication (`__mul__`)</strong>:
        <ul>
            <li><strong>Forward pass</strong>: For two scalar values \(a\) and \(b\), their product \(p\) is \(p = a \times b\).</li>
            <li><strong>Backward pass</strong>: The gradient of \(p\) with respect to \(a\) is \(b\), and with respect to \(b\) is \(a\). This means that during backpropagation, each input's gradient is the product of the other input and the output's gradient.</li>
        </ul>
    </li>

    <li><strong>ReLU Activation (`relu`)</strong>:
        <ul>
            <li><strong>Forward pass</strong>: The ReLU function is defined as \(R(x) = \max(0, x)\). This function outputs \(x\) if \(x\) is positive and 0 otherwise.</li>
            <li><strong>Backward pass</strong>: The derivative of the ReLU function is 1 for \(x > 0\) and 0 for \(x \leq 0\). Thus, the gradient is propagated through the function only if the input is positive; otherwise, it stops.</li>
        </ul>
    </li>
</ul>

<h3>Conceptual Application in Neural Networks</h3>

<ul>
    <li><strong>Addition and Multiplication</strong>: These operations are ubiquitous in neural networks, forming the basis of computing weighted sums of inputs in the neurons.</li>
    <li><strong>ReLU Activation</strong>: Commonly used as an activation function in neural networks due to its simplicity and effectiveness in introducing non-linearity, making learning complex patterns possible.</li>
</ul>

Understanding these operations and their implications on gradient flow is crucial for designing and training effective neural network models. By implementing these from scratch, one gains deeper insights into the workings of more sophisticated deep learning libraries.
''',
        'starter_code': """class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        # Implement addition here
        pass

    def __mul__(self, other):
        # Implement multiplication here
        pass

    def relu(self):
        # Implement ReLU here
        pass

    def backward(self):
        # Implement backward pass here
        pass""",
        'solution': """
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
""",
        'test_cases': [
            {
                "test": """a = Value(2);b = Value(3);c = Value(10);d = a + b * c  ;e = Value(7) * Value(2);f = e + d;g = f.relu()  
g.backward()
print(a,b,c,d,e,f,g)
""",
                "expected_output": """ Value(data=2, grad=1) Value(data=3, grad=10) Value(data=10, grad=3) Value(data=32, grad=1) Value(data=14, grad=1) Value(data=46, grad=1) Value(data=46, grad=1)"""
            }
        ],
        'class':True
    }
}

