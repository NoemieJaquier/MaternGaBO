"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu
"""

import math
import torch


def gegenbauer_polynomial(n, alpha, z):
    """
    This function computes the Gegenbauer polynomial C_n^alpha(z).

    Parameters
    ----------
    :param n: Gegenbauer polynomial parameter n
    :param alpha: Gegenbauer polynomial parameter alpha
    :param z: Gegenbauer polynomial function input

    Returns
    -------
    :return: Gegenbauer polynomial C_n^alpha(z)

    """
    # Initialization
    polynomial = 0.
    gamma_alpha = math.gamma(alpha)
    # Computes the summation serie
    for i in range(math.floor(n/2)+1):
        polynomial += math.pow(-1, i) * torch.pow(2*z, n-2*i) \
                      * (math.gamma(n-i+alpha) / (gamma_alpha * math.factorial(i) * math.factorial(n-2*i)))

        # Compute math.gamma(n - i + alpha)/math.gamma(alpha) = prod_{k=1:n-i}(n-i+alpha-k)
        # gamma_quotient = 1.
        # for k in range(n-i):
        #     gamma_quotient *= (n-i+alpha-(k+1))
        # # Update polynomial
        # polynomial += math.pow(-1, i) * torch.pow(2 * z, n - 2 * i) \
        #               * (gamma_quotient/ (math.factorial(i) * math.factorial(n - 2 * i)))

    return polynomial

