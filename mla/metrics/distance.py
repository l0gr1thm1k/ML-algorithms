import numpy as np

from math import sqrt


def euclidean_distance(a, b):
    """
    Calculate the straight line distance on a flat plane between two points.

    :param a: a vector of points.
    :param b: a vector of points.
    :return: a floating point value representative of the distance between the two vectors.
    """
    if isinstance(a, list) and isinstance(b, list):
        a = np.array(a)
        b = np.array(b)
        
    return sqrt(sum((a - b) ** 2))


def l2_distance(X):
    """
    Calculate the l2 normalization of the input matrix X.

    :param X: the matrix whose L2 distance we need to calculate.
    :return: a np.matrix of l2 distance for the input matrix X
    """
    sum_X = np.sum(X * X, axis=1)
    return (-2 * np.dot(X, X.T) + sum_X).T + sum_X