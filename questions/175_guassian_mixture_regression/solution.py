# ---------------------------------------- utf-8 encoding ---------------------------------
# This file contains Gaussian Process implementation.
import numpy as np
import math
from scipy.spatial.distance import euclidean
from scipy.special import kv as bessel_kv
from scipy.special import gamma
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit, softmax


# --- KERNEL FUNCTIONS --------------------------------------------------------
def matern_kernel(x: np.ndarray, x_prime: np.ndarray, length_scale=1.0, nu=1.5):
    d = euclidean(x, x_prime)
    if d == 0:
        return 1.0  # Covariance with self is 1 before scaling
    if nu == 0.5:
        return np.exp(-d / length_scale)
    elif nu == 1.5:
        return (1 + np.sqrt(3) * d / length_scale) * np.exp(
            -np.sqrt(3) * d / length_scale
        )
    elif nu == 2.5:
        return (
            1 + np.sqrt(5) * d / length_scale + 5 * d**2 / (3 * length_scale**2)
        ) * np.exp(-np.sqrt(5) * d / length_scale)
    else:
        factor = (2 ** (1 - nu)) / gamma(nu)
        scaled_d = np.sqrt(2 * nu) * d / length_scale
        return factor * (scaled_d**nu) * bessel_kv(nu, scaled_d)


def rbf_kernel(x: np.ndarray, x_prime, sigma=1.0, length_scale=1.0):
    # This is a squared exponential kernel

    # Calculate the squared euclidean distance
    sq_norm = np.linalg.norm(x - x_prime) ** 2

    # Correctly implement the formula
    return sigma**2 * np.exp(-sq_norm / (2 * length_scale**2))


def periodic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, period=1.0
):
    return sigma**2 * np.exp(
        -2 * np.sin(np.pi * np.linalg.norm(x - x_prime) / period) ** 2 / length_scale**2
    )


def linear_kernel(x: np.ndarray, x_prime: np.ndarray, sigma_b=1.0, sigma_v=1.0):
    return sigma_b**2 + sigma_v**2 * np.dot(x, x_prime)


def rational_quadratic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, alpha=1.0
):
    return sigma**2 * (
        1 + np.linalg.norm(x - x_prime) ** 2 / (2 * alpha * length_scale**2)
    ) ** (-alpha)


# --- BASE CLASS -------------------------------------------------------------


class _GaussianProcessBase:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None):
        self.kernel_name = kernel
        self.noise = noise
        self.kernel_params = kernel_params if kernel_params else {}
        self.X_train = None
        self.y_train = None
        self.K = None

    def _select_kernel(self, x1, x2):
        """Selects and computes the kernel value for two single data points."""
        if self.kernel_name == "rbf":
            return rbf_kernel(x1, x2, **self.kernel_params)
        elif self.kernel_name == "matern":
            return matern_kernel(x1, x2, **self.kernel_params)
        elif self.kernel_name == "periodic":
            return periodic_kernel(x1, x2, **self.kernel_params)
        elif self.kernel_name == "linear":
            return linear_kernel(x1, x2, **self.kernel_params)
        elif self.kernel_name == "rational_quadratic":
            return rational_quadratic_kernel(x1, x2, **self.kernel_params)
        else:
            raise ValueError(
                "Unsupported kernel. Choose from ['rbf', 'matern', 'periodic', 'linear', 'rational_quadratic']."
            )

    def _compute_covariance(self, X1, X2):
        """
        Computes the covariance matrix between two sets of points.
        This method fixes the vectorization bug from the original code.
        """
        # Ensuring X1 and X2 are 2D arrays
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        n1, _ = X1.shape
        n2, _ = X2.shape
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._select_kernel(X1[i], X2[j])
        return K


# --- REGRESSION MODEL -------------------------------------------------------
class GaussianProcessRegression(_GaussianProcessBase):
    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.K = self._compute_covariance(
            self.X_train, self.X_train
        ) + self.noise * np.eye(len(self.X_train))

        # Compute Cholesky decomposition for stable inversion
        self.L = cholesky(self.K, lower=True)
        # alpha = K_inv * y
        self.alpha = solve_triangular(
            self.L.T, solve_triangular(self.L, self.y_train, lower=True)
        )

    def predict(self, X_test, return_std=False):
        X_test = np.atleast_2d(X_test)
        K_s = self._compute_covariance(self.X_train, X_test)
        K_ss = self._compute_covariance(X_test, X_test)

        # Compute predictive mean
        mu = K_s.T @ self.alpha

        # Compute predictive variance
        v = solve_triangular(self.L, K_s, lower=True)
        cov = K_ss - v.T @ v

        if return_std:
            return mu, np.sqrt(np.diag(cov))
        return mu

    def log_marginal_likelihood(self):
        return (
            -0.5 * (self.y_train.T @ self.alpha)
            - np.sum(np.log(np.diag(self.L)))
            - len(self.X_train) / 2 * np.log(2 * np.pi)
        )

    def optimize_hyperparameters(self):
        # NOTE: This is a simplified optimizer for 'rbf' kernel's params.
        def objective(params):
            self.kernel_params = {
                "length_scale": np.exp(params[0]),
                "sigma": np.exp(params[1]),
            }
            self.fit(self.X_train, self.y_train)
            return -self.log_marginal_likelihood()

        init_params = np.log(
            [
                self.kernel_params.get("length_scale", 1.0),
                self.kernel_params.get("sigma", 1.0),
            ]
        )
        res = minimize(
            objective, init_params, method="L-BFGS-B", bounds=[(-5, 5), (-5, 5)]
        )

        self.kernel_params = {
            "length_scale": np.exp(res.x[0]),
            "sigma": np.exp(res.x[1]),
        }
        # Re-fit with optimal hyperparameters
        self.fit(self.X_train, self.y_train)
        print("Optimized Hyperparameters:", self.kernel_params)


if __name__ == "__main__":
    gp = GaussianProcessRegression(
        kernel="linear", kernel_params={"sigma_b": 0.0, "sigma_v": 1.0}, noise=1e-8
    )
    X_train = np.array([[1], [2], [4]])
    y_train = np.array([3, 5, 9])
    gp.fit(X_train, y_train)
    X_test = np.array([[3.0]])
    mu = gp.predict(X_test)
    print(f"{mu[0]:.4f}")
