import autograd.numpy as np
from autograd import elementwise_grad

from mla.NeuralNet.activations import get_activation
from mla.NeuralNet.parameters import Parameters

np.random.seed(9999)


class Layer(object):

    def setup(self, X_shape):
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

