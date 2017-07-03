from ..metrics import mse, mae, log_loss, hinge_loss, binary_cross_entropy

categorical_cross_entropy = log_loss()


def get_loss(name):
    """
    Get a loss function by name.

    :param name: name of the function as a string.
    :return: a loss function.
    """
    try:
        return globals()[name]
    except:
        raise ValueError('No loss function exists: %s' % name)