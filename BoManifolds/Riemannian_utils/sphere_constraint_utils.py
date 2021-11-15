"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier and Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
"""

import numpy as np


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
