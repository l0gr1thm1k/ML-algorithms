import autograd.numpy as np
from autograd import elementwise_grad

from mla.NeuralNet.activations import get_activation
from mla.NeuralNet.parameters import Parameters

np.random.seed(9999)


class Layer(object):

    def setup(self, x_shape):
        """Allocates initial weights."""
        pass

    def forward_pass(self, x):
        raise NotImplementedError()

    def backward_pass(self, delta):
        raise NotImplementedError()
    
    def shape(self, x_shape):
        """Returns the shape of the current layer."""
        raise NotImplementedError()
    

class ParamMixin(object):
    
    @property
    def parameters(self):
        return self._params
    
    
class PhaseMixin(object):
    _train = False
    
    @property
    def is_training(self):
        return self._train
    
    @is_training.setter
    def is_training(self, is_train=True):
        self._train = is_train
        
    @property
    def is_testing(self):
        return not self._train

    @is_testing.setter
    def is_testing(self, is_test=True):
        self._train = not is_test


class Dense(Layer, ParamMixin):

    def __init__(self, output_dim, parameters=None):
        """
        A fully connected layer.

        :param output_dim: an integer of output dimensions.
        :param parameters: default None
        """
        self._params = parameters
        self.output_dim = output_dim
        self.last_input = None

        if parameters is None:
            self._params = Parameters()

    def setup(self, x_shape):
        self._params.setup_weights(x_shape[1], self.output_dim)

    def forward_pass(self, x):
        self.last_input = x
        return self.weight(x)

    def weight(self, x):
        weight = np.dot(x, self._params['W'])
        return weight + self._params['b']

    def backward_pass(self, delta):
        delta_weight = np.dot(self.last_input.T, delta)
        delta_bias = np.sum(delta, axis=0)

        # update gradient values
        self._params.update_grad('W', delta_weight)
        self._params.update_grad('b', delta_bias)
        return np.dot(delta, self._params['W'].T)

    def shape(self, x_shape):
        return x_shape[0], self.output_dim


class Activation(Layer):

    def __init__(self, name):
        self.last_input = None
        self.activation = get_activation(name)
        # derivative of activation function
        self.activation_derivative = elementwise_grad(self.activation)

    def forward_pass(self, x):
        self.last_input = x
        return self.activation(x)

    def backward_pass(self, delta):
        return self.activation_derivative(self.last_input) * delta

    def shape(self, x_shape):
        return x_shape


class Dropout(Layer, PhaseMixin):
    """Randomly set a number of 'p' inputs to 0 at each training update."""

    def __init__(self, p=0.1):
        self.p = p
        self._mask = None

    def forward_pass(self, x):
        assert self.p > 0
        if self.is_training:
            self._mask = np.random.uniform(size=x.shape) > self.p
            y = x * self._mask
        else:
            y = x * (1.0 - self.p)
        return y

    def backward_pass(self, delta):
        return delta * self._mask

    def shape(self, x_shape):
        return x_shape


class TimeStepSlicer(Layer):
    """Take a specific time step from a 3D tensor."""

    def __init__(self, step=-1):
        self.step = step

    def forward_pass(self, x):
        return x[:, self.step, :]

    def backward_pass(self, delta):
        return np.repeat(delta[:, np.newaxis, :], 2, 1)

    def shape(self, x_shape):
        return x_shape[0], x_shape[2]


class TimeDistributedDense(Layer):
    """Apply regular Dense layer to every time step."""

    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.n_time_steps = None
        self.Dense = None
        self.input_dim = None

    def setup(self, x_shape):
        self.dense = Dense(self.output_dim)
        self.dense.setup(x_shape[0], x_shape[2])
        self.input_dim = x_shape[2]

    def forward_pass(self, x):
        n_timesteps = x.shape[1]
        x = x.reshape(-1, x.shape[-1])
        y = self.Dense.forward_pass(x)
        y = y.reshape((-1, n_timesteps, self.input_dim))
        return y

    @property
    def parameters(self):
        return self.dense._params

    def shape(self, x_shape):
        return x_shape[0], x_shape[1], self.output_dim