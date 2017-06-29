import numpy as np

EPS = 1e-15


def unhot(function):
    """
    Convert a one hot representation into a column.

    :param function: a function whose parameters actual and predicted are modified into columns.
    :return: return the generated wrapper function to modify the function parameters.
    """
    def wrapper(actual, predicted):
        if len(actual.shape) > 1 and actual.shape[1] > 1:
            actual = actual.argmax(axis=1)
        if len(predicted.shape) > 1 and predicted.shape[1] > 1:
            predicted = predicted.argmax(axis=1)
        return function(actual, predicted)
    return wrapper


def absolute_error(actual, predicted):
    """
    Calculate the absolute error on a set of data.

    :param actual: a vector or matrix of values.
    :param predicted: a vector or matrix of predicted values.
    :return: return a similarly sized vector or matrix with the absolute error between actual and predicted.
    """
    return np.abs(actual - predicted)


@unhot
def classification_error(actual, predicted):
    """
    Get the classification error on a set of values and associated predicted values.

    :param actual: a vector or matrix of gold standard values.
    :param predicted: a vector or matrix of predicted values.
    :return: a float value of the percentage of instances incorrectly classified.
    """
    return (actual != predicted).sum() / float(actual.shape[0])


@unhot
def accuracy(actual, predicted):
    """
    Get the classification accuracy over a set of gold and predicted values.

    :param actual: a vector or matrix of gold standard values.
    :param predicted: a vector or matrix of gold standard values.
    :return: a float value of the percentage of instances correctly classified.
    """
    return 1.0 - classification_error(actual, predicted)


def mean_absolute_error(actual, predicted):
    """
    Calculate the mean absolute error of a pairing of values

    :param actual: a vector or matrix of gold standard values.
    :param predicted: a vector or matrix of predicted values.
    :return: a float value representing the mean of the absolute errors
    """
    return np.mean(absolute_error(actual, predicted))


def squared_error(actual, predicted):
    """
    Calculate the squared error for each pair of actual and predicted values.
    :param actual: a vector or matrix of predicted values.
    :param predicted: a vector or matrix of gold standard values.
    :return: a vector or matrix of squared error values.
    """
    return (actual - predicted) ** 2


def mean_squared_error(actual, predicted):
    """
    Calculate the mean of squared errors over a set of actual and predicted values.

    :param actual: a vector or matrix of gold standard values.
    :param predicted: a vector or matrix of predicted values.
    :return: a float representing the mean of the square of errors.
    """
    return np.mean(squared_error(actual, predicted))


def root_mean_squared_error(actual, predicted):
    """
    Calculate the Root Mean Squared Error over a set of actual and predicted values.

    :param actual: a vector or matrix of predicted values.
    :param predicted: a vector or matrix of gold standard values.
    :return: the square root of the mean squared errors, a float.
    """
    return np.sqrt(mean_squared_error(actual, predicted))


def squared_log_error(actual, predicted):
    """
    Calculate the squared errors of the natural logarithms of values over sets of actual and predicted data points.

    :param actual: a vector or matrix of gold standard values.
    :param predicted: a vector or matrix of predicted values.
    :return: a vector or matrix of squared errors of the natural logarithms of the values.
    """
    # the +1 scales the vector by 1
    return (np.log(np.array(actual) + 1) - np.log(np.array(predicted) + 1)) ** 2


def mean_squared_log_error(actual, predicted):
    """
    Calculate the average of the squared log errors of values.

    :param actual: a vector or matrix of predicted values.
    :param predicted: a vector or matrix of gold standard values.
    :return: the average of squared log errors, a float value.
    """
    return np.mean(squared_log_error(actual, predicted))


def root_mean_squared_log_error(actual, predicted):
    """
    Get the square root of the average of squared log errors.

    :param actual: a vector or matrix of gold standard values.
    :param predicted: a vector or matrix of predicted values.
    :return: a float representing the square root of the mean of squared log errors.
    """
    return np.sqrt(root_mean_squared_log_error(actual, predicted))

