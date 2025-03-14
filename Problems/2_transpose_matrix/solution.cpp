#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cassert>

using namespace std;
using namespace Eigen;

//Transposes a given matrix using Eigen.
template <typename T>
Matrix<T, Dynamic, Dynamic> transpose_matrix(const Matrix<T, Dynamic, Dynamic>& a) {
    // If the matrix is empty, return an empty matrix
    if (a.size() == 0) {
        return Matrix<T, Dynamic, Dynamic>(0, 0);
    }
    return a.transpose();
}

//Unit tests for transpose_matrix function.
void test_transpose_matrix() {
    // Test case 1: Empty matrix (0x0)
    Matrix<int, Dynamic, Dynamic> emptyMat(0, 0);
    assert(transpose_matrix(emptyMat).size() == 0);

    // Test case 2: Degenerate matrix (1 row, 0 columns)
    Matrix<int, Dynamic, Dynamic> degenerateMat(1, 0);
    assert(transpose_matrix(degenerateMat).size() == 0);

    // Test case 3: Valid matrix transpose
    Matrix<int, Dynamic, Dynamic> a(2, 3);
    a << 1, 2, 3,
         4, 5, 6;

    Matrix<int, Dynamic, Dynamic> expected(3, 2);
    expected << 1, 4,
                2, 5,
                3, 6;

    assert(transpose_matrix(a) == expected);
}

int main() {
    test_transpose_matrix();
    cout << "All tests passed." << endl;
    return 0;
}