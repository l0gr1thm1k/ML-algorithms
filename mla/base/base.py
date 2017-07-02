import numpy as np


class BaseEstimator(object):

    x = None
    y = None
    y_required = True
    fit_required = True

    def _setup_input(self, x, y=None):
        """

        :param x:
        :param y:
        :return:
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if x.size == 0:
            raise ValueError('Number of feature must be > 0')

        if x.ndim == 1:
            self.n_samples, self.n_features = 1, x.shape
        else:
            self.n_samples, self.n_feaures = x.shape[0], np.prod(x.shape[1:])

        self.x = x

        if self.y_required:
            if y is None:
                raise ValueError('Missed required argument')

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError('Number of targets must be > 0')

        self.y = y

    def fit(self, x, y=None):
        self._setup_input(x, y)

    def predict(self, x=None):
        """

        :param x:
        :return:
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if self.x is not None or self.fit_required:
            return self._predict(x)
        else:
            raise ValueError("You must call 'fit' before 'predict'")

    def _predict(self, x=None):
        raise NotImplementedError()
