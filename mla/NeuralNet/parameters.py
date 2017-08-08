import numpy as np

from mla.NeuralNet.initializations import get_initializer

class Parameter(object):

    def __init__(self, init='glorot_uniform', scale=0.5, bias=1.0, regularizers=None, constraints=None):
        """
        A container for a layer's parameters

        :param init: default value is 'glorot_uniform'. This isn the name of the weight initialization unction.
        :param scale: float. default value is 0.5
        :param bias: float. default value is 1.0, the initial value for the bias
        :param regularizers: dict of weight regularizers.
        :param constrains: dict of weight contraints.
        """
        if constraints is None:
            self.constraints = {}
        else:
            self.constraints = constraints

        if regularizers is None:
            self.regularizers = {}
        else:
            self.regularizers = regularizers

        self.initial_bias = bias
        self.scale = scale
        self.init = get_initializer(init)

        self._params = {}
        self._grads = {}

    def setup_weights(self, W_shape, b_shape=None):
        if 'W' not in self._params:
            self._params['W'] = self.init(shape=W_shape, scale=self.scale)
            if b_shape is None:
                self._params['b'] = np.full(W_shape[1], self.initial_bias)
            else:
                self._params['b'] = np.full(b_shape, self.initial_bias)
        self.init_grad()

    def init_grad(self):
        """
        Initialize gradient arrays corresponding to each weight array
        """
        for key in self._param.keys():
            if key not in self._grads:
                self._grads[key] = np.zeros_like(self._params[key])

    def step(self, name, step):
        """
        Increase specific weight by amount of the step parameter

        :param name: the parameter name.
        :param step: the amount to increment the parameter by. step size.
        """
        self._params[name] += step

        if name in self.constraints:
            self._grads[name] = self.constraints[name].clip(self._params[name])

    def update_grad(self, name, value):
        """
        Update gradient values.

        :param name:
        :param value:
        :return:
        """
        self._grads[name] = value

        if name in self.regularizers:
            self._grads[name] += self.regularizers[name](self._params[name])

    @property
    def n_params(self):
        """
        Count the number of parameters in this layer
        """
        return sum([np.prod(self._params[x].shape) for x in self._params.keys()])

    def keys(self):
        return self._grads

    @property
    def grad(self):
        return self._grads

    # Allow access to the fields using dict syntax, e.g. parameters['W']
    def __getitem__(self, item):
        if item in self._params:
            return self._params[item]
        else:
            raise ValueError