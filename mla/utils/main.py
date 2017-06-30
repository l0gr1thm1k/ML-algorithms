import numpy as np


def one_hot(y):
    """
    Create a one hot matrix from a vector of values.

    :param y: a vector of integer values representing the index at which the value 1 is inserted .
    :return: a one hot matrix with n fields.
    """
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


def batch_iterator(x, batch_size=64):
    """
    Split data x into equal sized chunks.

    :param x: an iterable input.
    :param batch_size: the size to split a batch into.
    :return: yield iterable of size batch_size
    """
    n_samples = x.shape[0]
    n_batches = n_samples // batch_size
    batch_end = 0

    for b in range(n_batches):
        batch_begin = b * batch_size          # our beginning index for this batch
        batch_end = batch_begin + batch_size  # our end index for this batch

        x_batch = x[batch_begin:batch_end]

        yield x_batch

    if n_batches * batch_size < n_samples:
        yield x[batch_end:]
