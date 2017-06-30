import numpy as np


def check_data(a, b):
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
    def wrapper(a, b):
        a, b = check_data(a, b)
        return function(a, b)
    
    return wrapper