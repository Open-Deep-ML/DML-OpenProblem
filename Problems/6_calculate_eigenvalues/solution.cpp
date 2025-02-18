#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace Eigen;

// Templated function to calculate eigenvalues for a 2x2 matrix.
// Assumes that the matrix has real eigenvalues.
template<typename T>
vector<T> calculate_eigenvalues(const Matrix<T, 2, 2>& matrix) {
    // Use EigenSolver to compute eigenvalues.
    EigenSolver<Matrix<T, 2, 2>> solver(matrix);
    vector<T> eigenvals;

    // Extract the real parts of the eigenvalues.
    for (int i = 0; i < solver.eigenvalues().size(); ++i) {
        eigenvals.push_back(solver.eigenvalues()(i).real());
    }

    // Sort the eigenvalues in descending order.
    sort(eigenvals.begin(), eigenvals.end(), greater<T>());
    return eigenvals;
}

// Unit tests for the templated calculate_eigenvalues function.
void test_calculate_eigenvalues() {
    const double eps = 1e-6;  // tolerance for floating point comparisons

    // Test case 1: using double type
    Matrix2d matrix1;
    matrix1 << 2, 1,
               1, 2;
    vector<double> expected1 = {3.0, 1.0};
    vector<double> eigenvals1 = calculate_eigenvalues(matrix1);
    assert(fabs(eigenvals1[0] - expected1[0]) < eps);
    assert(fabs(eigenvals1[1] - expected1[1]) < eps);

    // Test case 2: using float type
    Matrix<float, 2, 2> matrix2;
    matrix2 << 4.0f, -2.0f,
               1.0f,  1.0f;
    vector<float> expected2 = {3.0f, 2.0f};
    vector<float> eigenvals2 = calculate_eigenvalues(matrix2);
    assert(fabs(eigenvals2[0] - expected2[0]) < eps);
    assert(fabs(eigenvals2[1] - expected2[1]) < eps);
}

int main() {
    test_calculate_eigenvalues();
    cout << "All calculate_eigenvalues tests passed." << endl;
    return 0;
}