import numpy as np
# import autograd.numpy as np

'''
reference:
https://en.wikipedia.org/wiki/Activation_function
'''

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(z))


def softmax(z):

    e = np.exp(z - np.amax(z, axis=1))
    return e / np.sum(e, axis=1)


def linear(z):
    return z


def relu(z):
    return np.maximum(0, z)


def softplus(z):
    """smoothed relu"""
    return np.log(1 + np.exp(z))


def softsign(z):
    return z / (1 + np.abs(z))


def tanh(z):
    return np.tanh(z)


def get_activation(name):
    """
    Get an activation function by name.

    :param name: the name of the function to return.
    :return: an activation function.
    """
    try:
        return globals()[name]
    except:
        raise ValueError("Invalid activation function name: %s" % name)
