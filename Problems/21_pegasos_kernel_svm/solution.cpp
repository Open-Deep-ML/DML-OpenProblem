#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using namespace std;
using namespace Eigen;

class PegasosKernelSVM {
private:
    string _kernel;
    double _lambda_val;
    size_t _iterations;
    double _sigma;
    MatrixXd kermat;
    void _compute_kernel_matrix(const MatrixXd& data);
public:
    VectorXd alphas;
    double b;
    PegasosKernelSVM(string, double, size_t, double);
    void fit(const MatrixXd&, const VectorXd&);
};

PegasosKernelSVM::PegasosKernelSVM(
    string kernel, double lambda_val, size_t iterations, double sigma=0.5
): _kernel(kernel), _lambda_val(lambda_val), _iterations(iterations), _sigma(sigma) {}

void PegasosKernelSVM::fit(const MatrixXd& data, const VectorXd& labels) {
    // Kernel matrix computation
    _compute_kernel_matrix(data);
    // Parameter initialization
    size_t n_samples = data.rows();
    alphas = VectorXd::Zero(n_samples);
    b = 0.0;
    // Pegasos iteration
    for (size_t t = 0; t < _iterations; t++) {
        double lr = 1.0 / (_lambda_val * (t + 1));
        for (size_t i = 0; i < n_samples; i++) {
            // Compute the kernelized prediction
            double pred = (kermat.row(i) * (alphas.array() * labels.array()).matrix()).sum() + b;
            // Update alphas and b
            if (labels(i) * pred < 1.0) {
                alphas(i) += lr * (labels(i) - _lambda_val * alphas(i));
                b += lr * labels(i);
            }
        }
    }
    return;
}

void PegasosKernelSVM::_compute_kernel_matrix(const MatrixXd& data) {
    size_t n_samples = data.rows();
    if (_kernel == "linear") {
        kermat = data * data.transpose();
    }
    else if (_kernel == "rbf") {
        kermat = MatrixXd::Zero(n_samples, n_samples);
        for (size_t i = 0; i < n_samples; i++) {
            kermat.row(i) = (
                - 0.5 * (data.rowwise() - data.row(i)).rowwise().squaredNorm() / (_sigma * _sigma)
            ).array().exp();
        }
    }
}

void test_pegasos_kernel_svm() {
    MatrixXd data(4, 2);
    data << 1, 2,
            2, 3,
            3, 1,
            4, 1;
    VectorXd labels(4);
    labels << 1, 1, -1, -1;
    VectorXd expected_alphas(4);
    double expected_b, tol = 1e-5;

    // Test case 1: Linear kernel
    PegasosKernelSVM linear_svm("linear", 0.01, 100);
    try {
        expected_alphas << 100.0, 0.0, -100.0, -100.0;
        expected_b = -937.4755;
        linear_svm.fit(data, labels);
        assert((linear_svm.alphas - expected_alphas).norm() < tol);
        assert(fabs(linear_svm.b - expected_b) < tol);
    } catch (const exception& e) {
        cout << "Test case 1 failed: " << e.what() << endl;
    }

    // Test case 2: RBF kernel
    PegasosKernelSVM rbf_svm("rbf", 0.01, 100, 0.5);
    try {
        expected_alphas << 100.0, 99.0, -100.0, -100.0;
        expected_b = -115.0;
        rbf_svm.fit(data, labels);
        assert((rbf_svm.alphas - expected_alphas).norm() < tol);
        assert(fabs(rbf_svm.b - expected_b) < tol);
    } catch (const exception& e) {
        cout << "Test case 2 failed: " << e.what() << endl;
    }
}

int main() {
    test_pegasos_kernel_svm();
    cout << "All pegasos_kernel_svm tests passed." << endl;
    return 0;
}