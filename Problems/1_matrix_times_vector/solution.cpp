#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cassert>

using namespace std;
using namespace Eigen;

// Template function that performs matrix-vector multiplication using Eigen.
template <typename T>
Matrix<T, Dynamic, 1> matrix_dot_vector(const Matrix<T, Dynamic, Dynamic>& a,
                                        const Matrix<T, Dynamic, 1>& b) 
{
    // Check if the matrix is empty or its column count differs from vector size
    if (a.rows() == 0 || a.cols() != b.size()) {
        // Return a vector of size 1 with value -1 to indicate error
        Matrix<T, Dynamic, 1> error(1);
        error(0) = static_cast<T>(-1);
        return error;
    }

    // Compute the matrix-vector product using Eigen's operator*
    return a * b;
}

// Unit test function for matrix_dot_vector
void test_matrix_dot_vector() {
    // 1. Invalid input test cases

    {
        // Both matrix and vector are empty
        Matrix<float, Dynamic, Dynamic> emptyMat(0, 0);
        Matrix<float, Dynamic, 1> emptyVec(0);
        auto res = matrix_dot_vector(emptyMat, emptyVec);
        // The returned vector should have exactly 1 element which is -1
        assert(res.size() == 1 && res(0) == -1);
    }
    {
        // Matrix is empty, vector has size(2)
        Matrix<float, Dynamic, Dynamic> emptyMat(0, 0);
        Matrix<float, Dynamic, 1> vec(2);
        vec << 1, 2;
        auto res = matrix_dot_vector(emptyMat, vec);
        assert(res.size() == 1 && res(0) == -1);
    }
    {
        // Matrix(1x2) is not empty, but vector is empty
        Matrix<float, Dynamic, Dynamic> mat(1, 2);
        mat << 1, 2;
        Matrix<float, Dynamic, 1> emptyVec(0);
        auto res = matrix_dot_vector(mat, emptyVec);
        assert(res.size() == 1 && res(0) == -1);
    }
    {
        // Matrix(2x2) but vector(1x1), dimension mismatch
        Matrix<float, Dynamic, Dynamic> mat(2, 2);
        mat << 1, 2,
               2, 4;
        Matrix<float, Dynamic, 1> vec(1);
        vec << 1;
        auto res = matrix_dot_vector(mat, vec);
        assert(res.size() == 1 && res(0) == -1);
    }

    // 2. Valid input test cases

    // Float type
    {
        Matrix<float, Dynamic, Dynamic> mat(2, 2);
        mat << 1, 2,
               2, 4;
        Matrix<float, Dynamic, 1> vec(2);
        vec << 1, 2;

        // The expected result should be [5, 10]
        auto res = matrix_dot_vector(mat, vec);
        Matrix<float, Dynamic, 1> expected(2);
        expected << 5.0f, 10.0f;

        // isApprox is used here for floating-point comparison (Eigen specific)
        assert(res.size() == expected.size() && res.isApprox(expected));
    }

    // Integer type
    {
        Matrix<int, Dynamic, Dynamic> mat(2, 3);
        mat << 1, 2, 3,
               2, 4, 6;
        Matrix<int, Dynamic, 1> vec(3);
        vec << 1, 2, 3;

        // The expected result should be [14, 28]
        auto res = matrix_dot_vector(mat, vec);
        Matrix<int, Dynamic, 1> expected(2);
        expected << 14, 28;

        // For integer types, direct equality checks are sufficient
        assert(res.size() == expected.size());
        for (int i = 0; i < res.size(); ++i) {
            assert(res(i) == expected(i));
        }
    }
}

int main() {
    // Run all tests to ensure correctness
    test_matrix_dot_vector();
    cout << "All tests passed." << endl;
    return 0;
}