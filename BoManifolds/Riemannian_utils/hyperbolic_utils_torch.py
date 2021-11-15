"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier and Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
"""

import torch


def lorentz_distance_torch(x1, x2, diag=False):
    """
    This function computes the Riemannian distance between points on a hyperbolic manifold.

    Parameters
    ----------
    :param x1: points on the hyperbolic manifold                                N1 x dim or b1 x ... x bk x N1 x dim
    :param x2: points on the hyperbolic manifold                                N2 x dim or b1 x ... x bk x N2 x dim

    Optional parameters
    -------------------
    :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.

    Returns
    -------
    :return: matrix of manifold distance between the points in x1 and x2         N1 x N2 or b1 x ... x bk x N1 x N2
    """
    if diag is False:
        # Expand dimensions to compute all vector-vector distances
        x1 = x1.unsqueeze(-2)
        x2 = x2.unsqueeze(-3)

        # Repeat x and y data along -2 and -3 dimensions to have b1 x ... x ndata_x x ndata_y x dim arrays
        x1 = torch.cat(x2.shape[-2] * [x1], dim=-2)
        x2 = torch.cat(x1.shape[-3] * [x2], dim=-3)

        # Difference between x1 and x2
        diff_x = x1.view(-1, x1.shape[-1]) - x2.view(-1, x2.shape[-1])

        # Compute the hyperbolic distance
        mink_inner_prod = inner_minkowski_columns(diff_x.transpose(-1, -2), diff_x.transpose(-1, -2))
        mink_sqnorms = torch.maximum(torch.zeros_like(mink_inner_prod), mink_inner_prod)
        mink_norms = torch.sqrt(mink_sqnorms + 1e-8)
        distance = 2 * torch.arcsinh(.5 * mink_norms).view(x1.shape[:-1])

    else:
        # Difference between x1 and x2
        diff_x = x1 - x2

        # Compute the hyperbolic distance
        mink_inner_prod = inner_minkowski_columns(diff_x.transpose(-1, -2), diff_x.transpose(-1, -2))
        mink_sqnorms = torch.maximum(torch.zeros_like(mink_inner_prod), mink_inner_prod)
        mink_norms = torch.sqrt(mink_sqnorms + 1e-8)
        distance = 2 * torch.arcsinh(.5 * mink_norms).squeeze(-1)

    return distance


def inner_minkowski_columns(x, y):
    return -x[0]*y[0] + torch.sum(x[1:]*y[1:], dim=0)


def from_poincare_to_lorentz(x):
    x_torch = torch.tensor(x)
    first_coord = 1 + torch.pow(torch.linalg.norm(x_torch), 2)
    lorentz_point = torch.cat((first_coord.reshape(1), 2*x_torch)) / (1 - torch.pow(torch.linalg.norm(x_torch), 2))
    return lorentz_point
