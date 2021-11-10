import numpy as np
'''
The functions of this file are based on the function of botorch (in botorch.optim).
'''


def norm_one_constraint(x):
    """
    This function defines an 1-norm equality constraint on the a vector.
    The value returned by the function is 0 if the equality constraints is satisfied.

    Parameters
    ----------
    :param x: vector

    Returns
    -------
    :return: difference between the norm of x and 1
    """
    return np.linalg.norm(x) - 1.
