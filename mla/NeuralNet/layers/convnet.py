import autograd.numpy as np

from mla.NeuralNet.layers import Layer, ParamMixin
from mla.NeuralNet.parameters import Parameters


class Convolution(Layer, ParamMixin):

    def __init__(self, n_filters=8, filter_shape=(3, 3), padding=(0, 0), stride=(1, 1), parameters=None):
        """
        A 2D convolutional layer.

        :param n_filters:
        :param filter_shape:
        :param padding:
        :param stride:
        :param parameters:
        """
        self.padding = padding
        self._params = parameters
        self.stride = stride
        self.filter_shape = filter_shape
        self.n_filters = n_filters
        if self._params is None:
            self._params = Parameters()

    def setup(self, x_shape):
        n_channels, self.height, self.width = x_shape[1:]

        w_shape = (self.n_filters, n_channels) + self.filter_shape
        b_shape = (self.n_filters)
        self._params.setup_weights(w_shape, b_shape)