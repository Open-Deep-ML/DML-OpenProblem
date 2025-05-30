# ------------------------------ utf-8 encoding -------------------------------

import numpy as np
from collections import defaultdict

class NaiveBayes():
    def __init__(self,  smoothing=1.0):
        self.smoothing = smoothing
        self.classes = None
        self.priors = None
        self.likelihoods = None

    def forward(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True) # extract classes and its freq
        self.priors = {
            cls: np.log(class_counts[i] / len(y)) for i, cls in enumerate(self.classes)
        }

        self._fit_gaussian(X , y)


    def _fit_gaussian(self, X, y):
        self.likelihoods = {}
        epsilon = 1e-9 

        for cls in self.classes:
            X_cls = X[y == cls]
            mean = np.mean(X_cls, axis=0)
            var = np.var(X_cls, axis=0) + self.smoothing + epsilon
            self.likelihoods[cls] = (mean, var)

    def _compute_posterior(self, sample): # golden heart of the algorithm (naive bayes)
        posteriors = {}

        for cls in self.classes:
            posterior = self.priors[cls]

            mean, var = self.likelihoods[cls]
            likelihood = -0.5 * np.sum(((sample - mean) ** 2) / var) - 0.5 * np.sum(np.log(2 * np.pi * var))

            posterior += likelihood
            posteriors[cls] = posterior

        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self._compute_posterior(sample) for sample in X])



def test_gaussian_naive_bayes():
    np.random.seed(42)

    # Simulated data: 2 classes, 3 features
    X_train = np.random.normal(0, 1, (100, 3))
    y_train = np.random.choice([0, 1], size=100)

    X_test = np.random.normal(0, 1, (10, 3))

    model = NaiveBayes( smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(X_test), "Prediction length mismatch!"
    assert set(y_pred).issubset({0, 1}), "Predictions should be binary (0 or 1)"
    print("Gaussian Naive Bayes test passed successfully.")


def test_gaussian_nb():
    np.random.seed(42)  # For reproducibility

    X = np.array([
        [5.1, 3.5],   # Class 0
        [4.9, 3.0],   # Class 0
        [7.0, 3.2],   # Class 1
        [6.4, 3.2],   # Class 1
        [5.9, 3.0]    # Class 1
    ])
    y = np.array([0, 0, 1, 1, 1])

    model = NaiveBayes()
    model.forward(X, y)

    X_test = np.array([[6.0, 3.1]])  # Should be class 1
    pred = model.predict(X_test)

    assert pred[0] == 1, f"Expected 1 but got {pred[0]}"
    print("Gaussian NB passed!")

def test_perfect_separation():
    # Clearly separable by feature 0
    X_train = np.array([
        [1.0, 2.0],
        [1.1, 2.2],
        [5.0, 3.0],
        [5.2, 3.1]
    ])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([
        [1.05, 2.1],  # Should be 0
        [5.1, 3.0]    # Should be 1
    ])

    model = NaiveBayes()
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert list(y_pred) == [0, 1], "Model failed to perfectly separate the classes"
    print("Test: Perfectly separable data passed.")


def test_imbalanced_classes():
    # Imbalanced: Class 0 dominates ( number of class 0 data points are more than other class in a good weightage)
    X_train = np.random.normal(0, 1, (95, 2))
    y_train = np.zeros(95)
    X_train = np.vstack([X_train, np.random.normal(5, 1, (5, 2))])
    y_train = np.concatenate([y_train, np.ones(5)])

    X_test = np.array([[5, 5]])  # closer to minority class (1)

    model = NaiveBayes()
    model.forward(X_train, y_train)
    pred = model.predict(X_test)

    assert pred[0] == 1, "Model ignored minority class prediction"
    print("Test: Imbalanced classes handled.")


def test_univariate_feature():
    # Only Single feature test
    X_train = np.array([[1], [2], [3], [10], [11], [12]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[2], [11]])

    model = NaiveBayes()
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert list(y_pred) == [0, 1], "Model failed on 1D Gaussian input"
    print("Test: Single feature Gaussian passed.")


def test_single_class_training():
    # All training samples from one class  (let us see what model does here)
    X_train = np.random.normal(0, 1, (10, 3))
    y_train = np.zeros(10)
    X_test = np.random.normal(0, 1, (3, 3))

    model = NaiveBayes()
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert all(p == 0 for p in y_pred), "Model trained on one class should always predict that class"
    print("Test: Single class training handled.")



if __name__ == "__main__":
    test_gaussian_naive_bayes()
    test_gaussian_nb()
    test_imbalanced_classes()
    test_perfect_separation()
    test_single_class_training()
    test_univariate_feature()
