#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

// Template function to compute the transpose of a matrix
template <typename T>
vector<vector<T>> transpose_matrix(const vector<vector<T>>& matrix) {
    // Check for empty matrix
    if (matrix.empty() || matrix[0].empty()) {
        return {{static_cast<T>(-1)}};  // Error code
    }

    int rows = matrix.size();
    int cols = matrix[0].size();

    // Check if all rows have the same number of columns (valid matrix structure)
    for (const auto& row : matrix) {
        if (row.size() != cols) {
            // Return error for malformed matrix
            return {{static_cast<T>(-1)}};  
        }
    }

    // Initialize the result matrix with the correct dimensions
    vector<vector<T>> res(cols, vector<T>(rows));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res[j][i] = matrix[i][j];
        }
    }

    return res;
}


// Unit tests for transpose_matrix function
void test_transpose_matrix() {
    // Valid integer matrix
    vector<vector<int>> int_matrix = {
        {1, 3, 5},
        {2, 4, 6}
    };
    vector<vector<int>> expected_int_result = {
        {1, 2},
        {3, 4},
        {5, 6}
    };
    assert(transpose_matrix(int_matrix) == expected_int_result);

    // Valid floating-point matrix
    vector<vector<double>> double_matrix = {
        {1.2, 3.4, 5.6},
        {2.1, 4.3, 6.5}
    };
    vector<vector<double>> expected_double_result = {
        {1.2, 2.1},
        {3.4, 4.3},
        {5.6, 6.5}
    };
    assert(transpose_matrix(double_matrix) == expected_double_result);

    // Edge case: Empty matrix
    assert(transpose_matrix<int>({}) == vector<vector<int>>{{-1}});

    // Edge case: Invalid matrix (row length mismatch)
    vector<vector<int>> malformed_matrix = {
        {1, 2},
        {3, 4, 5}
    };
    assert(transpose_matrix(malformed_matrix) == vector<vector<int>>{{-1}});

}

int main() {
    test_transpose_matrix();
    cout << "All tests passed." << endl;
    return 0;
}