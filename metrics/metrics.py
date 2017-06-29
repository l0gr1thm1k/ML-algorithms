import numpy as np

EPS = 1e-15


def unhot(function):
    """
    :desc: convert a one hot representation into a column
    :param function:
    :return:
    """
    def wrapper(actual, predicted):
        if len(actual.shape) > 1 and actual.shape[1] > 1:
            actual = actual.argmax(axis=1)
        if len(predicted.shape) > 1 and predicted.shape[1] > 1:
            predicted = predicted.argmax(axis=1)
        return function(actual, predicted)
    return wrapper


def absolute_error(actual, predicted):
    return np.abs(actual - predicted)
