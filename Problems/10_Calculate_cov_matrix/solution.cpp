#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

// Function to calculate the covariance matrix.
// The input 'data' is an (n_features x n_observations) matrix,
// where each row represents one feature and each column one observation.
MatrixXd calculate_covariance_matrix(const MatrixXd& data) {
    int n_features = data.rows();
    int n_observations = data.cols();
    MatrixXd cov(n_features, n_features);

    // Compute the mean for each feature (i.e., for each row)
    VectorXd means = data.rowwise().mean();

    // Calculate the covariance between each pair of features.
    // We take advantage of the symmetry of the covariance matrix.
    for (int i = 0; i < n_features; i++) {
        for (int j = i; j < n_features; j++) {
            double covariance = 0.0;
            for (int k = 0; k < n_observations; k++) {
                covariance += (data(i, k) - means(i)) * (data(j, k) - means(j));
            }
            covariance /= (n_observations - 1);
            cov(i, j) = covariance;
            cov(j, i) = covariance;  // The matrix is symmetric.
        }
    }
    return cov;
}

void test_calculate_covariance_matrix() {
    {
        // Test case 1: 2 features, 3 observations.
        MatrixXd data(2, 3);
        // Each row corresponds to a feature.
        data << 1, 2, 3,
                4, 5, 6;
        MatrixXd expected(2, 2);
        expected << 1.0, 1.0,
                    1.0, 1.0;
        MatrixXd cov = calculate_covariance_matrix(data);
        if (!cov.isApprox(expected, 1e-3)) {
            cout << "Test case 1 failed." << endl;
            cout << "Expected:" << endl << expected << endl;
            cout << "Got:" << endl << cov << endl;
            exit(1);
        }
    }
    {
        // Test case 2: 3 features, 3 observations.
        MatrixXd data(3, 3);
        data << 1, 5, 6,
                2, 3, 4,
                7, 8, 9;
        MatrixXd expected(3, 3);
        expected << 7.0, 2.5, 2.5,
                    2.5, 1.0, 1.0,
                    2.5, 1.0, 1.0;
        MatrixXd cov = calculate_covariance_matrix(data);
        if (!cov.isApprox(expected, 1e-3)) {
            cout << "Test case 2 failed." << endl;
            cout << "Expected:" << endl << expected << endl;
            cout << "Got:" << endl << cov << endl;
            exit(1);
        }
    }
    
}

int main() {
    test_calculate_covariance_matrix();
    cout << "All tests passed." << endl;
    return 0;
}