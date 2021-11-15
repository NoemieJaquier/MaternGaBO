"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier and Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
"""

import torch


def post_processing_init_sphere_torch(x):
    """
    This function post-processes vectors, so that its norm is equal to 1.

    Parameters
    ----------
    :param x: d-dimensional vectors [N x d]

    Returns
    -------
    :return: unit-norm vectors [N x d]

    """
    return x / torch.cat(x.shape[-1] * [torch.norm(x, dim=[-1]).unsqueeze(-1)], dim=-1)
