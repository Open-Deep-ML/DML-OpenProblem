#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

// Template function to perform matrix-vector multiplication
template <typename T>
vector<T> matrix_dot_vector(const vector<vector<T>>& a, const vector<T>& b) {
    // TODO: Implement matrix-vector multiplication
    
    // Return a vector of -1 if the matrix is not compatible with the vector
    if (a.empty() || a[0].size() != b.size()) {
        return vector<T>{-1};
    }

    // Initialize the result vector c
    vector<T> res(a.size(), 0);

    // Perform matrix-vector multiplication
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < a[i].size(); j++) {
            res[i] += a[i][j] * b[j];
        }
    }

    return res;
}

// Unit tests for the matrix_dot_vector function
void test_matrix_dot_vector() {
    // Test cases for invalid inputs
    assert((matrix_dot_vector<float>({}, {}) == vector<float>{-1}));
    assert((matrix_dot_vector<float>({}, {1, 2}) == vector<float>{-1}));
    assert((matrix_dot_vector<float>({{1, 2}}, {}) == vector<float>{-1}));
    assert((matrix_dot_vector<float>({{1, 2}, {2, 4}}, {1}) == vector<float>{-1}));

    // Test cases for valid float inputs
    vector<vector<float>> a = {{1, 2}, {2, 4}};
    vector<float> b = {1, 2};
    assert((matrix_dot_vector(a, b) == vector<float>{5, 10}));

    // Test cases for valid int inputs
    vector<vector<int>> a2 = {{1, 2, 3}, {2, 4, 6}};
    vector<int> b2 = {1, 2, 3};
    assert((matrix_dot_vector(a2, b2) == vector<int>{14, 28}));
}

// Main function
int main() {
    test_matrix_dot_vector();
    cout << "All tests passed." << endl;
    return 0;
}