import numpy as np
from mla.base import BaseEstimator

def softmax(z):
    # Avoid numerical overflow. This is duplicated code from neuralnet.activations.softmax
    # also uses autograd.np as np
    e = np.exp(z - np.amax(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


class NaiveBayesClassifier(BaseEstimator):
    """Gaussian Naive Bayes"""
    n_classes = 2

    def fit(self, x, y=None):
        self._setup_input(x, y)

        # Check target labels
        assert list(np.unique(y)) == [0, 1]

        # Mean and variance for each class and feature combination
        self._mean = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self._var = np.zeros((self.n_classes, self.n_features), dtype=np.float64)

        self._priors = np.zeros(self.n_classes, dtype=np.float64)

        for c in range(self.n_classes):
            # filter by class
            x_c = x[y == c]

            # calculate mean, prior and variabce for each class
            self._mean[c, :] = x_c.mean(axis=0)
            self._var[c, :] = x_c.var(axis=0)
            self._priors[c] = x_c.shape[0] / float(x.shape[0])

    def _predict(self, x=None):
        # Apply predict probability along each row
        predictions = np.apply_along_axis(self._predict_row, 1, x)

        # normalize probabilities so each row sums to 1.0
        return softmax(predictions)

    def _predict_row(self, x):
        """Predict log likeliehood of a row"""
        output = []
        for y in range(self.n_classes):
            prior = np.log(self._priors[y])
            posterior = np.log(self._pdf(y, x)).sum()
            prediction = prior + posterior

            output.append(prediction)
        return output

    def _pdf(self, n_class, x):
        """Calculate a Gaussian probability density function for each feature."""
        mean = self._mean[n_class]
        var = self._var[n_class]

        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)