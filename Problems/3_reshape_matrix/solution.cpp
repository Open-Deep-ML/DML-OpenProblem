#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cassert>

using namespace std;
using namespace Eigen;

// Reshapes a given matrix into a new shape using Eigen.

template <typename T>
Matrix<T, Dynamic, Dynamic> reshape_matrix(const Matrix<T, Dynamic, Dynamic>& a, int new_rows, int new_cols) {
    int rows = a.rows();
    int cols = a.cols();

    // Check if reshaping is feasible
    if (rows * cols != new_rows * new_cols) {
        return Matrix<T, Dynamic, Dynamic>(0, 0);  // Return an empty matrix if reshaping is invalid
    }

    // Flatten the matrix into a 1D vector
    vector<T> flat_matrix;
    flat_matrix.reserve(rows * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            flat_matrix.push_back(a(r, c));
        }
    }

    // Populate the new reshaped matrix
    Matrix<T, Dynamic, Dynamic> reshaped(new_rows, new_cols);
    int index = 0;
    for (int r = 0; r < new_rows; ++r) {
        for (int c = 0; c < new_cols; ++c) {
            reshaped(r, c) = flat_matrix[index++];
        }
    }

    return reshaped;
}

// Unit tests for reshape_matrix function.
void test_reshape_matrix() {
    // Test case 1: Valid reshape (2x4 -> 4x2)
    Matrix<int, Dynamic, Dynamic> a(2, 4);
    a << 1, 2, 3, 4,
         5, 6, 7, 8;
    Matrix<int, Dynamic, Dynamic> expected1(4, 2);
    expected1 << 1, 2,
                 3, 4,
                 5, 6,
                 7, 8;
    assert(reshape_matrix(a, 4, 2) == expected1);

    // Test case 2: Valid reshape (2x3 -> 3x2)
    Matrix<int, Dynamic, Dynamic> b(2, 3);
    b << 1, 2, 3,
         4, 5, 6;
    Matrix<int, Dynamic, Dynamic> expected2(3, 2);
    expected2 << 1, 2,
                 3, 4,
                 5, 6;
    assert(reshape_matrix(b, 3, 2) == expected2);

    // Test case 3: Invalid reshape (2x4 -> 1x4) (should return empty)
    assert(reshape_matrix(a, 1, 4).size() == 0);

    // Test case 4: Valid reshape (2x2 -> 4x1)
    Matrix<int, Dynamic, Dynamic> c(2, 2);
    c << 1, 2,
         3, 4;
    Matrix<int, Dynamic, Dynamic> expected3(4, 1);
    expected3 << 1,
                 2,
                 3,
                 4;
    assert(reshape_matrix(c, 4, 1) == expected3);

    // Test case 5: Valid reshape (3x2 -> 2x3)
    Matrix<int, Dynamic, Dynamic> d(3, 2);
    d << 1, 2,
         3, 4,
         5, 6;
    Matrix<int, Dynamic, Dynamic> expected4(2, 3);
    expected4 << 1, 2, 3,
                 4, 5, 6;
    assert(reshape_matrix(d, 2, 3) == expected4);

    // Test case 6: Valid reshape with floating-point numbers (3x3 -> 9x1)
    Matrix<double, Dynamic, Dynamic> e(3, 3);
    e << 1.5, 2.2, 3.1,
         4.7, 5.9, 6.3,
         7.7, 8.8, 9.9;
    Matrix<double, Dynamic, Dynamic> expected5(9, 1);
    expected5 << 1.5, 2.2, 3.1, 4.7, 5.9, 6.3, 7.7, 8.8, 9.9;
    assert(reshape_matrix(e, 9, 1).isApprox(expected5));

    // Test case 7: Invalid reshape (1x2 -> 2x2) (should return empty)
    Matrix<int, Dynamic, Dynamic> f(1, 2);
    f << 1, 2;
    assert(reshape_matrix(f, 2, 2).size() == 0);
}

int main() {
    test_reshape_matrix();
    cout << "All tests passed." << endl;
    return 0;
}