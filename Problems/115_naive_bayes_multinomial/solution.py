# ------------------------------ utf-8 encoding -------------------------------
# This is implementation of Multinomial Naive Bayes with all the test cases 

import numpy as np
from collections import defaultdict

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.classes = None
        self.priors = None
        self.likelihoods = None

    def forward(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True) # extract classes and its freq
        self.priors = {
            cls: np.log(class_counts[i] / len(y)) for i, cls in enumerate(self.classes)
        }

        
        self._fit_multinomial(X , y)
        
    def _fit_multinomial(self, X, y):
        self.likelihoods = {}
        for cls in self.classes:
            X_cls = X[y == cls]
            word_counts = np.sum(X_cls, axis=0) + self.smoothing
            self.likelihoods[cls] = np.log(word_counts / np.sum(word_counts))


    def _compute_posterior(self, sample): # golden heart of the algorithm (naive bayes)
        posteriors = {}

        for cls in self.classes:
            posterior = self.priors[cls]
            likelihood = np.sum(sample * self.likelihoods[cls])

            posterior += likelihood
            posteriors[cls] = posterior

        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self._compute_posterior(sample) for sample in X])




def test_multinomial_naive_bayes():
    np.random.seed(42)

    # Simulated multinomial data: 3 classes, 4 features (e.g., word counts)
    X_train = np.random.randint(1, 10, (100, 4))
    y_train = np.random.choice([0, 1, 2], size=100)

    X_test = np.random.randint(1, 10, (10, 4))

    model = NaiveBayes( smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(X_test), "Prediction length mismatch!"
    assert set(y_pred).issubset({0, 1, 2}), "Predictions should be in the range of [0, 1, 2]"
    print("Multinomial Naive Bayes test passed successfully.")




def test_two_classes_textlike_counts():
    # Simulated for binary classification
    X_train = np.array([
        [2, 1, 0, 0],
        [3, 0, 1, 0],
        [0, 2, 1, 1],
        [0, 1, 3, 2]
    ])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([
        [1, 0, 0, 0],  #  class 0
        [0, 1, 2, 1]   #  class 1
    ])

    model = NaiveBayes(smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert list(y_pred) == [0, 1], "Expected class predictions to match word patterns."
    print("Test 2: Binary class classification passed.")


def test_class_imbalance_multiclass():
    X_train = np.random.randint(1, 5, (30, 3))
    y_train = np.array([0]*24 + [1]*4 + [2]*2)  # highly imbalanced
    X_test = np.random.randint(1, 5, (5, 3))

    model = NaiveBayes(smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == 5
    assert set(y_pred).issubset({0, 1, 2}), "Predictions out of class range"
    print("Test 3: Multiclass imbalance handled successfully.")


def test_zero_features_row():
    # Model must handle rows with all zero counts (e.g., empty document)
    X_train = np.array([
        [2, 1, 1],
        [0, 3, 1],
        [1, 0, 2]
    ])
    y_train = np.array([0, 1, 1])
    X_test = np.array([
        [0, 0, 0]  # zero counts
    ])

    model = NaiveBayes(smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred[0] in [0, 1], "Model should still produce a valid prediction"
    print("Test 4: All-zero input row handled with smoothing.")


def test_single_class_only():
    X_train = np.random.randint(1, 10, (10, 4))
    y_train = np.zeros(10)  # all one class
    X_test = np.random.randint(1, 10, (3, 4))

    model = NaiveBayes(smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert np.all(y_pred == 0), "Model trained on one class should predict only that class"
    print("Test 5: Single class training scenario passed.")


def test_minimum_valid_case():
    X_train = np.array([[2]])
    y_train = np.array([1])
    X_test = np.array([[3]])

    model = NaiveBayes(smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred[0] == 1, "Minimum single-sample test failed"
    print("Test 6: Minimum valid training and prediction passed.")


if __name__ == "__main__":
    test_multinomial_naive_bayes()
    test_two_classes_textlike_counts()
    test_class_imbalance_multiclass()
    test_zero_features_row()
    test_single_class_only()
    test_minimum_valid_case()
