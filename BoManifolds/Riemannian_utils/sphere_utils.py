"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier and Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
"""

import numpy as np
import scipy as sc
import pymanopt.manifolds as pyman_man


def expmap(u, x0):
    """
    This function maps a vector u lying on the tangent space of x0 into the manifold.

    Parameters
    ----------
    :param u: vector in the tangent space
    :param x0: basis point of the tangent space

    Returns
    -------
    :return: x: point on the manifold
    """
    if np.ndim(x0) < 2:
        x0 = x0[:, None]

    if np.ndim(u) < 2:
        u = u[:, None]

    norm_u = np.sqrt(np.sum(u*u, axis=0))
    x = x0 * np.cos(norm_u) + u * np.sin(norm_u)/norm_u

    x[:, norm_u < 1e-16] = x0

    return x


def logmap(x, x0):
    """
    This functions maps a point lying on the manifold into the tangent space of a second point of the manifold.

    Parameters
    ----------
    :param x: point on the manifold
    :param x0: basis point of the tangent space where x will be mapped

    Returns
    -------
    :return: u: vector in the tangent space of x0
    """
    if np.ndim(x0) < 2:
        x0 = x0[:, None]

    if np.ndim(x) < 2:
        x = x[:, None]

    theta = np.arccos(np.maximum(np.minimum(np.dot(x0.T, x), 1.), -1.))
    u = (x - x0 * np.cos(theta)) * theta/np.sin(theta)

    u[:, theta[0] < 1e-16] = np.zeros((u.shape[0], 1))

    return u


def sphere_distance(x, y):
    """
    This function computes the Riemannian distance between two points on the manifold.

    Parameters
    ----------
    :param x: point on the manifold
    :param y: point on the manifold

    Returns
    -------
    :return: distance: manifold distance between x and y
    """
    if np.ndim(x) < 2:
        x = x[:, None]

    if np.ndim(y) < 2:
        y = y[:, None]

    # Compute the inner product (should be [-1,1])
    inner_product = np.dot(x.T, y)
    inner_product = np.max(np.min(inner_product, 1), -1)
    return np.arccos(inner_product)


def parallel_transport_operator(x1, x2):
    """
    This function computes the parallel transport operator from x1 to x2.
    Transported vectors can be computed as u.dot(v).

    Parameters
    ----------
    :param x1: point on the manifold
    :param x2: point on the manifold

    Returns
    -------
    :return: operator: parallel transport operator
    """
    if np.sum(x1-x2) == 0.:
        return np.eye(x1.shape[0])
    else:
        if np.ndim(x1) < 2:
            x1 = x1[:, None]

        if np.ndim(x2) < 2:
            x2 = x2[:, None]

        x_dir = logmap(x2, x1)
        norm_x_dir = np.sqrt(np.sum(x_dir*x_dir, axis=0))
        normalized_x_dir = x_dir / norm_x_dir
        u = np.dot(-x1 * np.sin(norm_x_dir), normalized_x_dir.T) + \
            np.dot(normalized_x_dir * np.cos(norm_x_dir), normalized_x_dir.T) + np.eye(x_dir.shape[0]) - \
            np.dot(normalized_x_dir, normalized_x_dir.T)

        return u


def karcher_mean_sphere(data, nb_iter=10):
    """
    This function computes the mean of points lying on the manifold (Fréchet/Karcher mean).

    Parameters
    ----------
    :param data: data points lying on the manifold

    Optional parameters
    -------------------
    :param nb_iter: number of iterations

    Returns
    -------
    :return: m: mean of the datapoints
    """
    # Initialize the mean as equal to the first datapoint
    m = data[:, 0]
    for i in range(nb_iter):
        data_tgt = logmap(data, m)
        m_tgt = np.mean(data_tgt, axis=1)
        m = expmap(m_tgt, m)

    return m


def get_axisangle(d):
    """
    Gets axis-angle representation of a point lying on a unit sphere
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param d: point on the sphere

    Returns
    -------
    :return: axis, angle: corresponding axis and angle representation
    """
    norm = np.sqrt(d[0]**2 + d[1]**2)
    if norm < 1e-6:
        return np.array([0, 0, 1]), 0
    else:
        vec = np.array([-d[1], d[0], 0])
        return vec/norm, np.arccos(d[2])


def rotation_from_sphere_points(x, y):
    """
    Gets the rotation matrix that moves x to y in the geodesic path on the sphere.
    Based on the equations of "Analysis of principal nested spheres", Sung et al. 2012 (appendix)

    Parameters
    ----------
    :param x: point on a sphere
    :param y: point on a sphere

    Returns
    -------
    :return: rotation matrix
    """
    if np.ndim(x) < 2:
        x = x[:, None]
    if np.ndim(y) < 2:
        y = y[:, None]

    dim = x.shape[0]

    in_prod = np.dot(x.T, y)
    in_prod = np.max(np.min(in_prod, 1), -1)
    c_vec = x - y * in_prod
    c_vec = c_vec / np.linalg.norm(c_vec)

    R = np.eye(dim) + np.sin(np.arccos(in_prod)) * (np.dot(y, c_vec.T) - np.dot(c_vec, y.T)) + (in_prod - 1.) * (np.dot(y, y.T) + np.dot(c_vec, c_vec.T))

    return R


def unit_sphere_to_rotation_matrix(unit_vector):
    """
    This function transforms a point lying on a sphere (i.e., unit vector) to a rotation matrix.
    This function is valid for rotation matrices of dimension 2 (from S1) and 3 (from S3).

    Parameters
    ----------
    :param unit_vector: a unit vector on S1 or S3

    Returns
    -------
    :return: a rotation matrix 2x2 or 3x3, or -1 if the dimension of the rotation matrix cannot be handled.
    """
    if np.ndim(unit_vector) < 2:
        unit_vector = unit_vector[:, None]

    if unit_vector.shape[0] == 4:
        return quaternion_to_rotation_matrix(unit_vector)
    elif unit_vector.shape[0] == 2:
        R = np.zeros((2, 2))
        R[:, 0] = unit_vector[:, 0]
        R[0, 1] = unit_vector[1]
        R[1, 1] = -unit_vector[0]
        # Ensure determinant 1
        if np.linalg.det(R) < 0:
            R = np.array([R[:, 1], R[:, 0]])
        return R
    else:
        raise NotImplementedError


def quaternion_to_rotation_matrix(q):
    """
    This function transforms a quaternion into a 3x3 rotation matrix.

    Parameters
    ----------
    :param quaternion: a quaternion or a batch of quaternion    N x [scalar term, vector term]

    Returns
    -------
    :return: 3x3 rotation matrices
    """

    if np.ndim(q) < 2:
        q = q[:, None]

    w, x, y, z = q[:, 0]
    R = np.array([[2 * (w ** 2 + x ** 2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 2 * (w ** 2 + y ** 2) - 1, 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w ** 2 + z ** 2) - 1]])
    return R

