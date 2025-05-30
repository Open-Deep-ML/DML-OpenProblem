# ------------------------------ utf-8 encoding -------------------------------
# This is implementation of Naive Bayes ml model of Bernauli varient with test cases
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

        self._fit_bernoulli(X , y)


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

            prob_1, prob_0 = self.likelihoods[cls]
            likelihood = np.sum(sample * prob_1 + (1 - sample) * prob_0)

            posterior += likelihood
            posteriors[cls] = posterior

        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self._compute_posterior(sample) for sample in X])



def test_bernoulli_naive_bayes():
    np.random.seed(42)

    # Simulated binary data: 2 classes, 5 features
    X_train = np.random.randint(0, 2, (100, 5))
    y_train = np.random.choice([0, 1], size=100)

    X_test = np.random.randint(0, 2, (10, 5))

    model = NaiveBayes(smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(X_test), "Prediction length mismatch!"
    assert set(y_pred).issubset({0, 1}), "Predictions should be binary (0 or 1)"  # must predict only 0 , 1
    print("Test 1: Basic test passed.")


def test_single_feature():
    X_train = np.array([[0], [1], [0], [1]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[0], [1]])

    model = NaiveBayes(smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert list(y_pred) == [0, 1], "Model should distinguish 0 and 1 correctly."   # model is disguished
    print("Test 2: Single feature classification passed.")


def test_unseen_feature_combination():
    X_train = np.array([[0, 0], [1, 0], [0, 1]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[1, 1]])  # Not seen in training 

    model = NaiveBayes(smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred[0] in [0, 1], "Model should make a prediction for unseen input."
    print("Test 3: Unseen feature combination handled with smoothing.")


def test_extreme_class_imbalance():
    X_train = np.random.randint(0, 2, (100, 5))
    y_train = np.array([0] * 95 + [1] * 5)
    X_test = np.random.randint(0, 2, (5, 5))

    model = NaiveBayes(smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(X_test)
    print("Test 4: Class imbalance handled.")


def test_all_same_class():
    X_train = np.random.randint(0, 2, (10, 3))
    y_train = np.zeros(10)
    X_test = np.random.randint(0, 2, (3, 3))

    model = NaiveBayes(smoothing=1.0)
    model.forward(X_train, y_train)
    y_pred = model.predict(X_test)

    assert np.all(y_pred == 0), "If trained on one class, model should predict only that class."
    print("Test 5: Single class training passed.")

    
def test_binary_nb():
    np.random.seed(42)

    # Binary features (0 or 1)
    X = np.array([
        [1, 0, 1],   # Spam
        [1, 1, 0],   # Spam
        [0, 0, 1],   # Ham
        [0, 1, 0],   # Ham
        [1, 1, 1]    # Spam
    ])
    y = np.array([1, 1, 0, 0, 1])
    model = NaiveBayes()
    model.forward(X, y)

    X_test = np.array([[1, 0, 1]])  # Should be spam (1)
    pred = model.predict(X_test)

    assert pred[0] == 1, f"Expected 1 but got {pred[0]}"
    print("Test 6: Binary NB passed!")

if __name__ == "__main__":
    test_bernoulli_naive_bayes()
    test_single_feature()
    test_unseen_feature_combination()
    test_extreme_class_imbalance()
    test_all_same_class()
    test_binary_nb()
