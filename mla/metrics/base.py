import numpy as np


def check_data(a, b):
    """
    Verify that input data has the same type and shape, otherwise raise an exception.

    :param a: an input parameter. Should be an iterable.
    :param b: an input parameter. Should be an iterable.
    :return a: np.array of values with equivalent size to param b.
    :return b: np.array of values with equivalent size to param a.
    :raises ValueError: if types of a and b are non-equivalent or size of a and b are non-equivalent.
    """
    if not isinstance(a, np.array):
        a = np.array(a)
        
    if not isinstance(b, np.array):
        b = np.array(b)
        
    if type(a) != type(b):
        raise ValueError('Type Mismatch: %s and %s' % (type(a), type(b)))
        
    if a.size != b.size:
        raise ValueError('Arrays must be equal length')
        
    return a, b


def validate_input(function):
    """
    Wrap a function with the check_data function.

    :param function: The function to wrap.
    :return: the wrapper function with validated data.
    """
    def wrapper(a, b):
        a, b = check_data(a, b)
        return function(a, b)
    
    return wrapper