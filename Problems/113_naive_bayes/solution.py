# ------------------------------ utf-8 encoding -------------------------------
# Optimized Naive Bayes Implementation (Gaussian, Multinomial, Bernoulli)

import numpy as np
from collections import defaultdict

class NaiveBayes():
    def __init__(self, model_type="gaussian", smoothing=1.0):
        self.model_type = model_type.lower()
        self.smoothing = smoothing
        self.classes = None
        self.priors = None
        self.likelihoods = None

    def forward(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True) # extract classes and its freq
        self.priors = {
            cls: np.log(class_counts[i] / len(y)) for i, cls in enumerate(self.classes)
        }

        fit_methods = {
            "gaussian": self._fit_gaussian,
            "multinomial": self._fit_multinomial,
            "bernoulli": self._fit_bernoulli
        }

        if self.model_type not in fit_methods:
            # making it similar to sklearn
            raise ValueError("Unsupported model type. Choose from 'gaussian', 'multinomial', or 'bernoulli'")

        fit_methods[self.model_type](X, y)

    def _fit_gaussian(self, X, y):
        self.likelihoods = {}
        epsilon = 1e-9 

        for cls in self.classes:
            X_cls = X[y == cls]
            mean = np.mean(X_cls, axis=0)
            var = np.var(X_cls, axis=0) + self.smoothing + epsilon
            self.likelihoods[cls] = (mean, var)

    def _fit_multinomial(self, X, y):
        self.likelihoods = {}
        for cls in self.classes:
            X_cls = X[y == cls]
            word_counts = np.sum(X_cls, axis=0) + self.smoothing
            self.likelihoods[cls] = np.log(word_counts / np.sum(word_counts))

    def _fit_bernoulli(self, X, y):
        self.likelihoods = {}
        for cls in self.classes:
            X_cls = X[y == cls]
            prob = (np.sum(X_cls, axis=0) + self.smoothing) / (X_cls.shape[0] + 2 * self.smoothing)
            self.likelihoods[cls] = (np.log(prob), np.log(1 - prob))

    def _compute_posterior(self, sample): # golden heart of the algorithm (naive bayes)
        posteriors = {}

        for cls in self.classes:
            posterior = self.priors[cls]

            if self.model_type == "gaussian":
                mean, var = self.likelihoods[cls]
                likelihood = -0.5 * np.sum(((sample - mean) ** 2) / var) - 0.5 * np.sum(np.log(2 * np.pi * var))

            elif self.model_type == "multinomial":
                likelihood = np.sum(sample * self.likelihoods[cls])

            elif self.model_type == "bernoulli":
                prob_1, prob_0 = self.likelihoods[cls]
                likelihood = np.sum(sample * prob_1 + (1 - sample) * prob_0)

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

    model = NaiveBayes(model_type="gaussian", smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(X_test), "Prediction length mismatch!"
    assert set(y_pred).issubset({0, 1}), "Predictions should be binary (0 or 1)"
    print("Gaussian Naive Bayes test passed successfully.")

def test_multinomial_naive_bayes():
    np.random.seed(42)

    # Simulated multinomial data: 3 classes, 4 features (e.g., word counts)
    X_train = np.random.randint(1, 10, (100, 4))
    y_train = np.random.choice([0, 1, 2], size=100)

    X_test = np.random.randint(1, 10, (10, 4))

    model = NaiveBayes(model_type="multinomial", smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(X_test), "Prediction length mismatch!"
    assert set(y_pred).issubset({0, 1, 2}), "Predictions should be in the range of [0, 1, 2]"
    print("Multinomial Naive Bayes test passed successfully.")

def test_bernoulli_naive_bayes():
    np.random.seed(42)

    # Simulated binary data: 2 classes, 5 features
    X_train = np.random.randint(0, 2, (100, 5))
    y_train = np.random.choice([0, 1], size=100)

    X_test = np.random.randint(0, 2, (10, 5))

    model = NaiveBayes(model_type="bernoulli", smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(X_test), "Prediction length mismatch!"
    assert set(y_pred).issubset({0, 1}), "Predictions should be binary (0 or 1)"
    print("Bernoulli Naive Bayes test passed successfully.")

if __name__ == "__main__":
    test_bernoulli_naive_bayes()
    test_gaussian_naive_bayes()
    test_multinomial_naive_bayes()