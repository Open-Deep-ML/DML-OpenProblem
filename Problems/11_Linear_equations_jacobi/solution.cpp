#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

// Jacobi solver: approximate solution after a given number of iterations.
VectorXd solve_jacobi(const MatrixXd &A, const VectorXd &b, int iterations) {
  int n = A.rows();
  VectorXd d = A.diagonal();
  // Subtract diagonal part (converted to dense) from A.
  MatrixXd offDiag = A - d.asDiagonal().toDenseMatrix();
  VectorXd x = VectorXd::Zero(n), x_new = x;

  for (int iter = 0; iter < iterations; iter++) {
    for (int i = 0; i < n; i++) {
      double sum = (offDiag.row(i) * x).sum();
      x_new(i) = (b(i) - sum) / d(i);
    }
    x = x_new;
  }

  // Round results to 4 decimal places.
  for (int i = 0; i < n; i++)
    x(i) = round(x(i) * 10000.0) / 10000.0;

  return x;
}

void test_solve_jacobi() {
  {
    MatrixXd A(3, 3);
    A << 5, -2, 3, -3, 9, 1, 2, -1, -7;
    VectorXd b(3);
    b << -1, 2, 3;
    VectorXd expected(3);
    expected << 0.146, 0.2032, -0.5175;
    if (!solve_jacobi(A, b, 2).isApprox(expected, 1e-3)) {
      cout << "Test 1 failed." << endl;
      exit(1);
    }
  }
  {
    MatrixXd A(3, 3);
    A << 4, 1, 2, 1, 5, 1, 2, 1, 3;
    VectorXd b(3);
    b << 4, 6, 7;
    VectorXd expected(3);
    expected << -0.0806, 0.9324, 2.4422;
    if (!solve_jacobi(A, b, 5).isApprox(expected, 1e-3)) {
      cout << "Test 2 failed." << endl;
      exit(1);
    }
  }
  {
    MatrixXd A(3, 3);
    A << 4, 2, -2, 1, -3, -1, 3, -1, 4;
    VectorXd b(3);
    b << 0, 7, 5;
    VectorXd expected(3);
    expected << 1.7083, -1.9583, -0.7812;
    if (!solve_jacobi(A, b, 3).isApprox(expected, 1e-3)) {
      cout << "Test 3 failed." << endl;
      exit(1);
    }
  }
}

int main() {
  test_solve_jacobi();
  cout << "All tests passed." << endl;
  return 0;
}