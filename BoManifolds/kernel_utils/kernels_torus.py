"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu
"""

import torch
import gpytorch

from BoManifolds.kernel_utils.kernels_sphere import CircleRiemannianGaussianKernel, \
    CircleRiemannianIntegratedMaternKernel


class TorusProductOfManifoldsRiemannianGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on a torus by considering
    it as a product of circle manifolds.

    Attributes
    ----------
    self.dim, dimension of the torus manifold on which the data handled by the kernel are living
    self.torus_kernel, product of circle kernels

    Methods
    -------
    forward(point1_on_torus, point2_on_torus, diagonal_matrix_flag=False, **params):

    """
    def __init__(self, dim, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the torus manifold on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(TorusProductOfManifoldsRiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None,
                                                                              **kwargs)

        # Dimension of the torus
        self.dim = dim

        # Initialise the product of kernels
        kernels = [CircleRiemannianGaussianKernel(active_dims=torch.tensor(list(range(2*i, 2*i+2))))
                   for i in range(self.dim)]

        self.torus_kernel = gpytorch.kernels.ProductKernel(*kernels)

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a torus manifold by considering it
        as a product of circle manifolds

        Parameters
        ----------
        :param x1: input points on the torus
        :param x2: input points on the torus

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # If the points are given as angles, transform them into coordinates on circles
        if x1.shape[-1] == self.dim:
            x1_circles = torch.zeros(list(x1.shape[:-1]) + [2 * self.dim], dtype=x1.dtype)
            x1_circles[..., ::2] = torch.cos(x1)
            x1_circles[..., 1::2] = torch.sin(x1)
        else:
            x1_circles = x1
        if x2.shape[-1] == self.dim:
            x2_circles = torch.zeros(list(x2.shape[:-1]) + [2 * self.dim], dtype=x2.dtype)
            x2_circles[..., ::2] = torch.cos(x2)
            x2_circles[..., 1::2] = torch.sin(x2)
        else:
            x2_circles = x2

        # Kernel
        kernel = self.torus_kernel.forward(x1_circles, x2_circles)
        return kernel


class TorusProductOfManifoldsRiemannianMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on a torus by considering it as a
    product of circle manifolds.

    Attributes
    ----------
    self.dim, dimension of the torus manifold on which the data handled by the kernel are living
    self.torus_kernel, product of circle kernels

    Methods
    -------
    forward(point1_on_torus, point2_on_torus, diagonal_matrix_flag=False, **params):

    """
    def __init__(self, dim, nu=None, nu_prior=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the torus manifold on which the data handled by the kernel are living
        :param nu: smoothness parameter of the circle kernels (can be a list/tuple or a single parameter)
        :param nu_prior: smoothness prior of the circle kernels (can be a list/tuple or a single parameter)

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(TorusProductOfManifoldsRiemannianMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None,
                                                                            **kwargs)

        # Dimension of the torus
        self.dim = dim

        # Smoothness parameters in list or tuple format
        if not isinstance(nu, (list, tuple)):
            nu = [nu for i in range(self.dim)]
        if not isinstance(nu_prior, (list, tuple)):
            nu_prior = [nu_prior for i in range(self.dim)]

        # Initialise the product of kernels
        kernels = [CircleRiemannianIntegratedMaternKernel(nu=nu[i], nu_prior=nu_prior[i],
                                                          active_dims=torch.tensor(list(range(2*i, 2*i+2))))
                   for i in range(self.dim)]

        self.torus_kernel = gpytorch.kernels.ProductKernel(*kernels)

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Matérn kernel matrix between inputs x1 and x2 belonging to a torus manifold by considering it
        as a product of circle manifolds

        Parameters
        ----------
        :param x1: input points on the torus
        :param x2: input points on the torus

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # If the points are given as angles, transform them into coordinates on circles
        if x1.shape[-1] == self.dim:
            x1_circles = torch.zeros(list(x1.shape[:-1]) + [2*self.dim], dtype=x1.dtype)
            x1_circles[..., ::2] = torch.cos(x1)
            x1_circles[..., 1::2] = torch.sin(x1)
        else:
            x1_circles = x1
        if x2.shape[-1] == self.dim:
            x2_circles = torch.zeros(list(x2.shape[:-1]) + [2*self.dim], dtype=x2.dtype)
            x2_circles[..., ::2] = torch.cos(x2)
            x2_circles[..., 1::2] = torch.sin(x2)
        else:
            x2_circles = x2

        # Kernel
        kernel = self.torus_kernel.forward(x1_circles, x2_circles)
        return kernel
