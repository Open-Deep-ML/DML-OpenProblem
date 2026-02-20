import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.classes = None
        self.priors = None
        self.likelihoods = None

    def forward(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.priors = {cls: np.log(class_counts[i] / len(y)) for i, cls in enumerate(self.classes)}
        self.likelihoods = {}
        for cls in self.classes:
            X_cls = X[y == cls]
            prob = (np.sum(X_cls, axis=0) + self.smoothing) / (X_cls.shape[0] + 2 * self.smoothing)
            self.likelihoods[cls] = (np.log(prob), np.log(1 - prob))

    def _compute_posterior(self, sample):
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
