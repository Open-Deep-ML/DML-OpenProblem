#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cassert>

using namespace std;
using namespace Eigen;

// Computes the mean of a matrix along rows or columns.

template <typename T>
vector<T> calculate_matrix_mean(const Matrix<T, Dynamic, Dynamic>& matrix, const string& mode) {
    if (matrix.size() == 0) {
        return {};  // Return an empty vector for an empty matrix
    }

    if (mode == "column") {
        auto means = matrix.colwise().mean().eval();  // Force evaluation
        return vector<T>(means.data(), means.data() + means.size());
    } 
    else if (mode == "row") {
        auto means = matrix.rowwise().mean().eval();  // Force evaluation
        return vector<T>(means.data(), means.data() + means.size());
    } 
    else {
        throw invalid_argument("Mode must be 'row' or 'column'");
    }
}

//brief Unit tests for calculate_matrix_mean function.
void test_calculate_matrix_mean() {
    // Test case 1: Column-wise mean (int)
    Matrix<int, Dynamic, Dynamic> matrix1(3, 3);
    matrix1 << 1, 2, 3,
               4, 5, 6,
               7, 8, 9;
    vector<int> expected1 = {4, 5, 6};  // Integer division keeps integer result
    assert(calculate_matrix_mean(matrix1, "column") == expected1);

    // Test case 2: Row-wise mean (int)
    vector<int> expected2 = {2, 5, 8};
    assert(calculate_matrix_mean(matrix1, "row") == expected2);

    // Test case 3: Column-wise mean (double)
    Matrix<double, Dynamic, Dynamic> matrix2(3, 2);
    matrix2 << 1.5, 2.5,
               3.5, 4.5,
               5.5, 6.5;
    vector<double> expected3 = {3.5, 4.5};
    assert(calculate_matrix_mean(matrix2, "column") == expected3);

    // Test case 4: Row-wise mean (double)
    vector<double> expected4 = {2.0, 4.0, 6.0};
    assert(calculate_matrix_mean(matrix2, "row") == expected4);

    // Test case 5: Empty matrix
    Matrix<int, Dynamic, Dynamic> emptyMatrix(0, 0);
    assert(calculate_matrix_mean(emptyMatrix, "row").empty());
    assert(calculate_matrix_mean(emptyMatrix, "column").empty());
}

int main() {
    test_calculate_matrix_mean();
    cout << "All tests passed." << endl;
    return 0;
}