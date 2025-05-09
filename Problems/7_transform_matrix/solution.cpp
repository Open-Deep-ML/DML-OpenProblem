#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>

using namespace Eigen;
using namespace std;

// Transform matrix A using transformation matrices T and S.
// If either T or S is non-invertible, return a 1x1 matrix with the single element -1.
MatrixXd transform_matrix(const MatrixXd &A, const MatrixXd &T, const MatrixXd &S) {
    // Check if T and S are invertible (using a small threshold to account for floating-point precision)
    if (fabs(T.determinant()) < 1e-9 || fabs(S.determinant()) < 1e-9) {
        MatrixXd error(1, 1);
        error(0, 0) = -1;
        return error;
    }

    // Compute the inverse of T
    MatrixXd T_inv = T.inverse();

    // Perform the matrix transformation: T_inv * A * S
    MatrixXd result = T_inv * A * S;

    return result;
}

void test_transform_matrix() {
    {
        // Test case 1
        MatrixXd A(2, 2), T(2, 2), S(2, 2);
        A << 1, 2,
             3, 4;
        T << 2, 0,
             0, 2;
        S << 1, 1,
             0, 1;
        MatrixXd result = transform_matrix(A, T, S);
        MatrixXd expected(2, 2);
        expected << 0.5, 1.5,
                    1.5, 3.5;
        if (!result.isApprox(expected, 1e-3)) {
            cout << "Test case 1 failed.\n";
            cout << "Result:\n" << result << "\nExpected:\n" << expected << "\n";
            exit(1);
        }
    }

    {
        // Test case 2
        MatrixXd A(2, 2), T(2, 2), S(2, 2);
        A << 1, 0,
             0, 1;
        T << 1, 2,
             3, 4;
        S << 2, 0,
             0, 2;
        MatrixXd result = transform_matrix(A, T, S);
        MatrixXd expected(2, 2);
        expected << -4.0, 2.0,
                     3.0, -1.0;
        if (!result.isApprox(expected, 1e-3)) {
            cout << "Test case 2 failed.\n";
            cout << "Result:\n" << result << "\nExpected:\n" << expected << "\n";
            exit(1);
        }
    }

    {
        // Test case 3
        MatrixXd A(2, 2), T(2, 2), S(2, 2);
        A << 2, 3,
             1, 4;
        T << 3, 0,
             0, 3;
        S << 1, 1,
             0, 1;
        MatrixXd result = transform_matrix(A, T, S);
        MatrixXd expected(2, 2);
        expected << 0.667, 1.667,
                    0.333, 1.667;
        if (!result.isApprox(expected, 1e-3)) {
            cout << "Test case 3 failed.\n";
            cout << "Result:\n" << result << "\nExpected:\n" << expected << "\n";
            exit(1);
        }
    }

    {
        // Test case 4: S is non-invertible; expected result is a 1x1 matrix with -1.
        MatrixXd A(2, 2), T(2, 2), S(2, 2);
        A << 2, 3,
             1, 4;
        T << 3, 0,
             0, 3;
        S << 1, 1,
             1, 1;  // This matrix is singular.
        MatrixXd result = transform_matrix(A, T, S);
        if (result.rows() != 1 || result.cols() != 1 || result(0, 0) != -1) {
            cout << "Test case 4 failed.\n";
            cout << "Result:\n" << result << "\nExpected: [-1]\n";
            exit(1);
        }
    }
}

int main() {
    test_transform_matrix();
    cout << "All transform_matrix tests passed.\n";
    return 0;
}