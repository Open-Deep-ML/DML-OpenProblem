#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

// Function to multiply two matrices.
// If the number of columns in 'a' does not equal the number of rows in 'b',
// return a 1×1 matrix with the single element -1.
MatrixXd matrixmul(const MatrixXd &a, const MatrixXd &b) {
    if (a.cols() != b.rows()) {
        MatrixXd error(1, 1);
        error(0, 0) = -1;
        return error;
    }
    return a * b;
}

void test_matrixmul() {
    {
        // Test case 1
        MatrixXd a(3, 3);
        a << 1, 2, 3,
             2, 3, 4,
             5, 6, 7;
        MatrixXd b(3, 3);
        b << 3, 2, 1,
             4, 3, 2,
             5, 4, 3;
        MatrixXd expected(3, 3);
        expected << 26, 20, 14,
                    38, 29, 20,
                    74, 56, 38;
        MatrixXd result = matrixmul(a, b);
        if (!result.isApprox(expected)) {
            cout << "Test case 1 failed." << endl;
            cout << "Result:\n" << result << endl;
            cout << "Expected:\n" << expected << endl;
            exit(1);
        }
    }

    {
        // Test case 2
        MatrixXd a(3, 2);
        a << 0, 0,
             2, 4,
             1, 2;
        MatrixXd b(2, 2);
        b << 0, 0,
             2, 4;
        MatrixXd expected(3, 2);
        expected << 0, 0,
                    8, 16,
                    4, 8;
        MatrixXd result = matrixmul(a, b);
        if (!result.isApprox(expected)) {
            cout << "Test case 2 failed." << endl;
            cout << "Result:\n" << result << endl;
            cout << "Expected:\n" << expected << endl;
            exit(1);
        }
    }

    {
        // Test case 3: Dimension mismatch.
        // Here, 'a' is a 3×2 matrix and 'b' is a 3×3 matrix, so multiplication is invalid.
        // Expected output is a 1×1 matrix with the value -1.
        MatrixXd a(3, 2);
        a << 0, 0,
             2, 4,
             1, 2;
        MatrixXd b(3, 3);
        b << 0, 0, 1,
             2, 4, 1,
             1, 2, 3;
        MatrixXd result = matrixmul(a, b);
        if (!(result.rows() == 1 && result.cols() == 1 && result(0, 0) == -1)) {
            cout << "Test case 3 failed." << endl;
            cout << "Result:\n" << result << endl;
            cout << "Expected: A 1×1 matrix with -1." << endl;
            exit(1);
        }
    }

}

int main() {
    test_matrixmul();
    cout << "All matrixmul tests passed." << endl;
    return 0;
}