#include <cassert>
#include <eigen3/Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

// Multiplies each element of the matrix by the given scalar.
template <typename T>
Matrix<T, Dynamic, Dynamic>
scalar_multiply(const Matrix<T, Dynamic, Dynamic> &matrix, T scalar) {
  return matrix * scalar;
}

// Unit tests for the scalar_multiply function.
void test_scalar_multiply() {
  // Test case 1: 2x2 integer matrix multiplied by 2.
  Matrix<int, Dynamic, Dynamic> matrix1(2, 2);
  matrix1 << 1, 2, 3, 4;
  Matrix<int, Dynamic, Dynamic> expected1(2, 2);
  expected1 << 2, 4, 6, 8;
  assert(scalar_multiply(matrix1, 2) == expected1);

  // Test case 2: 2x2 integer matrix multiplied by -1.
  Matrix<int, Dynamic, Dynamic> matrix2(2, 2);
  matrix2 << 0, -1, 1, 0;
  Matrix<int, Dynamic, Dynamic> expected2(2, 2);
  expected2 << 0, 1, -1, 0;
  assert(scalar_multiply(matrix2, -1) == expected2);
}

int main() {
  test_scalar_multiply();
  cout << "All tests passed." << endl;
  return 0;
}