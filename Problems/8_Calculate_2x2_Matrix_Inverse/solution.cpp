#include <iostream>
#include <optional>
#include <eigen3/Eigen/Dense>
#include <cmath>

using namespace Eigen;
using namespace std;

// Compute the inverse of a 2x2 matrix.
// It accepts a 2x2 Eigen matrix of any type T and returns its inverse wrapped in std::optional.
// If the matrix is non-invertible, it returns std::nullopt.
template <typename T>
std::optional<Matrix<T, 2, 2>> inverse_2x2(const Matrix<T, 2, 2>& matrix) {
    // Extract elements from the matrix
    T a = matrix(0, 0);
    T b = matrix(0, 1);
    T c = matrix(1, 0);
    T d = matrix(1, 1);

    // Calculate the determinant
    T determinant = a * d - b * c;

    // If the determinant is nearly zero, the matrix is non-invertible.
    if (std::abs(determinant) < static_cast<T>(1e-9)) {
        return std::nullopt;
    }

    // Compute the inverse using the formula for 2x2 matrices
    Matrix<T, 2, 2> inv;
    inv(0, 0) = d / determinant;
    inv(0, 1) = -b / determinant;
    inv(1, 0) = -c / determinant;
    inv(1, 1) = a / determinant;

    return inv;
}

// Test function for inverse_2x2
void test_inverse_2x2() {
    {
        // Test case 1 using double
        Matrix2d matrix;
        matrix << 4.0, 7.0,
                  2.0, 6.0;
        auto result = inverse_2x2(matrix);
        Matrix2d expected;
        expected << 0.6, -0.7,
                   -0.2,  0.4;
        if (!result.has_value() || !result.value().isApprox(expected, 1e-3)) {
            cout << "Test case 1 failed." << endl;
            if (result.has_value()) {
                cout << "Result:\n" << result.value() << "\nExpected:\n" << expected << endl;
            } else {
                cout << "Matrix is non-invertible." << endl;
            }
            exit(1);
        }
    }

    {
        // Test case 2 using double
        Matrix2d matrix;
        matrix << 2.0, 1.0,
                  6.0, 2.0;
        auto result = inverse_2x2(matrix);
        Matrix2d expected;
        expected << -1.0, 0.5,
                     3.0, -1.0;
        if (!result.has_value() || !result.value().isApprox(expected, 1e-3)) {
            cout << "Test case 2 failed." << endl;
            if (result.has_value()) {
                cout << "Result:\n" << result.value() << "\nExpected:\n" << expected << endl;
            } else {
                cout << "Matrix is non-invertible." << endl;
            }
            exit(1);
        }
    }

}

int main() {
    test_inverse_2x2();
    cout << "All inverse_2x2 tests passed." << endl;
    return 0;
}