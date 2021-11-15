"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier and Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
"""

import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'


def sphere_distance_torch(x1, x2, diag=False):
    """
    This function computes the Riemannian distance between points on a sphere manifold.

    Parameters
    ----------
    :param x1: points on the sphere                                             N1 x dim or b1 x ... x bk x N1 x dim
    :param x2: points on the sphere                                             N2 x dim or b1 x ... x bk x N2 x dim

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

        # Expand dimension to perform inner product
        x1 = x1.unsqueeze(-2)
        x2 = x2.unsqueeze(-1)

        # Compute the inner product (should be [-1,1])
        inner_product = torch.bmm(x1.view(-1, 1, x1.shape[-1]), x2.view(-1, x2.shape[-2], 1)).view(x1.shape[:-2])

    else:
        # Expand dimensions to compute all vector-vector distances
        x1 = x1.unsqueeze(-1).transpose(-1, -2)
        x2 = x2.unsqueeze(-1)
        inner_product = torch.bmm(x1, x2).squeeze(-1)

    # Clamp in case any value is not in the interval [-1,1]
    # A small number is added/substracted to the bounds to avoid NaNs during backward computation.
    inner_product = inner_product.clamp(-1.+1e-15, 1.-1e-15)

    return torch.acos(inner_product)


def rotation_from_sphere_points_torch(x, y):
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
    if x.dim() == 1:
        x = x.unsqueeze(-2)
    if y.dim() == 1:
        y = y.unsqueeze(-2)

    dim = x.shape[1]

    # Compute the inner product
    inner_product = torch.mm(x, y.T)
    # Clamp in case any value is not in the interval [-1,1]
    # A small number is added/substracted to the bounds to avoid NaNs during backward computation.
    inner_product = inner_product.clamp(-1. + 1e-15, 1. - 1e-15)

    # Compute intermediate vector
    c_vec = x - y * inner_product
    c_vec = c_vec / torch.norm(c_vec)

    R = torch.eye(dim, dim, dtype=inner_product.dtype) + \
        torch.sin(torch.acos(inner_product)) * (torch.mm(y.T, c_vec) - torch.mm(c_vec.T, y)) + \
        (inner_product - 1.) * (torch.mm(y.T, y) + torch.mm(c_vec.T, c_vec))

    return R


def rotation_matrix_to_unit_sphere_torch(rotation_matrix):
    """
    This function transforms a rotation matrix to a point lying on a sphere (i.e., unit vector).
    This function is valid for rotation matrices of dimension 2 (to S1) and 3 (to S3).

    Parameters
    ----------
    :param R: rotation matrix

    Returns
    -------
    :return: a unit vector on S1 or S3, or -1 if the dimension of the rotation matrix cannot be handled.
    """
    if rotation_matrix.shape[-1] == 3:
        return rotation_matrix_to_quaternion(rotation_matrix)
    elif rotation_matrix.shape[-1] == 2:
        init_shape = list(rotation_matrix.shape)
        new_shape = init_shape[:-2]
        new_shape.append(2)
        R = rotation_matrix.view(-1, init_shape[-2], init_shape[-1])
        return (R[:, 0]).view(new_shape)
    else:
        raise NotImplementedError


def rotation_matrix_to_quaternion(rotation_matrix):
    """
    This function transforms a 3x3 rotation matrix into a quaternion.
    This function was implemented based on Peter Corke's robotics toolbox.

    Parameters
    ----------
    :param R: 3x3 rotation matrix

    Returns
    -------
    :return: a quaternion [scalar term, vector term]
    """
    init_shape = list(rotation_matrix.shape)
    R = rotation_matrix.view(-1,  init_shape[-2], init_shape[-1])

    qs = torch.minimum(torch.sqrt(R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) + 1)/2.0, torch.ones(R.shape[0], dtype=R.dtype).to(device))
    kx = R[:, 2, 1] - R[:, 1, 2]   # Oz - Ay
    ky = R[:, 0, 2] - R[:, 2, 0]   # Ax - Nz
    kz = R[:, 1, 0] - R[:, 0, 1]   # Ny - Ox

    q = torch.zeros((R.shape[0], 4), dtype=R.dtype).to(device)
    q[:, 0] = qs

    for i in range(R.shape[0]):
        if (R[i, 0, 0] >= R[i, 1, 1]) and (R[i, 0, 0] >= R[i, 2, 2]):
            kx1 = R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2] + 1  # Nx - Oy - Az + 1
            ky1 = R[i, 1, 0] + R[i, 0, 1]                # Ny + Ox
            kz1 = R[i, 2, 0] + R[i, 0, 2]                # Nz + Ax
            add = (kx[i] >= 0)
        elif (R[i, 1, 1] >= R[i, 2, 2]):
            kx1 = R[i, 1, 0] + R[i, 0, 1]                # Ny + Ox
            ky1 = R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2] + 1  # Oy - Nx - Az + 1
            kz1 = R[i, 2, 1] + R[i, 1, 2]                # Oz + Ay
            add = (ky[i] >= 0)
        else:
            kx1 = R[i, 2, 0] + R[i, 0, 2]                # Nz + Ax
            ky1 = R[i, 2, 1] + R[i, 1, 2]                # Oz + Ay
            kz1 = R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1] + 1  # Az - Nx - Oy + 1
            add = (kz[i] >= 0)

        if add:
            kxi = kx[i] + kx1
            kyi = ky[i] + ky1
            kzi = kz[i] + kz1
        else:
            kxi = kx[i] - kx1
            kyi = ky[i] - ky1
            kzi = kz[i] - kz1

        nm = torch.norm(torch.Tensor([kxi, kyi, kzi]))
        if not nm == 0:
            s = torch.sqrt(1 - qs[i]**2) / nm
            q[i, 1] = s*kxi
            q[i, 2] = s*kyi
            q[i, 3] = s*kzi

    new_shape = init_shape[:-2]
    new_shape.append(4)
    return q.view(new_shape)


def unit_sphere_to_rotation_matrix_torch(unit_vector):
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
    if unit_vector.shape[-1] == 4:
        return quaternion_to_rotation_matrix(unit_vector)
    elif unit_vector.shape[-1] == 2:
        init_shape = list(unit_vector.shape)
        new_shape = init_shape[:-1]
        new_shape.append(2)
        new_shape.append(2)
        r1 = unit_vector.view(-1, init_shape[-1])[:, :, None]
        r2 = torch.vstack((r1[:, 1], -r1[:, 0])).T[:, :, None]
        R = torch.cat((r1, r2), -1)
        return R.view(new_shape)
    else:
        raise NotImplementedError


def quaternion_to_rotation_matrix(quaternion):
    """
    This function transforms a quaternion into a 3x3 rotation matrix.

    Parameters
    ----------
    :param quaternion: a quaternion or a batch of quaternion    N x [scalar term, vector term]

    Returns
    -------
    :return: 3x3 rotation matrices
    """

    init_shape = list(quaternion.shape)
    q = quaternion.view(-1, init_shape[-1])
    R = torch.zeros((quaternion.shape[0], 3, 3), dtype=quaternion.dtype).to(device)

    for i in range(R.shape[0]):
        w, x, y, z = q[i]
        R[i] = torch.tensor([[2 * (w ** 2 + x ** 2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                             [2 * (x * y + w * z), 2 * (w ** 2 + y ** 2) - 1, 2 * (y * z - w * x)],
                             [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w ** 2 + z ** 2) - 1]],
                            dtype=quaternion.dtype)

    new_shape = init_shape[:-1]
    new_shape.append(3)
    new_shape.append(3)
    return R.view(new_shape)


if __name__ == '__main__':
    x_s = torch.rand((1, 4))
    x_s /= torch.norm(x_s)
    x_so = unit_sphere_to_rotation_matrix_torch(x_s)
    x_s2 = rotation_matrix_to_unit_sphere_torch(x_so)

    x_s = torch.rand((1, 2))
    x_s /= torch.norm(x_s)
    x_so = unit_sphere_to_rotation_matrix_torch(x_s)
    x_s2 = rotation_matrix_to_unit_sphere_torch(x_so)


