"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu
"""

import numpy as np
import torch
import gpytorch
from gpytorch.constraints import Positive


if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'


class EuclideanHeatKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Heat covariance matrix between input points on the Euclidean manifold.

    Attributes
    ----------
    self.dim, dimension of the Euclidean space R^d on which the data handled by the kernel are living

    Methods
    -------
    forward(point(s)1, point(s)2, t, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the Euclidean manifold R^d on which the data handled by the kernel are living
        """

        self.has_lengthscale = False
        super(EuclideanHeatKernel, self).__init__(has_lengthscale=False, ard_num_dims=None, **kwargs)

    def forward(self, x1, x2, t, diag=False, **params):
        """
        Computes the Heat kernel matrix between inputs x1 and x2 belonging to a Euclidean manifold.

        Parameters
        ----------
        :param x1: input points
        :param x2: input points
        :param t: time

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2 at time t
        """
        # Data dimension
        dim = x1.shape[1]

        # Compute distance
        squared_distance = self.covar_dist(x1, x2, diag=diag, square_dist=True)
        exp_component = torch.exp(- squared_distance / (4*t))

        # Kernel
        kernel = exp_component / torch.pow(torch.tensor(4*np.pi*t), dim/2)
        return kernel


def unnormalized_heat_kernel(dist_sq, lengthscale):
    return torch.exp(-dist_sq/(2*torch.pow(lengthscale, 2)))


def link_function(dist_sq, t, lengthscale, smoothness, dimension):
    l, nu, d = lengthscale, smoothness, dimension

    result = torch.pow(t, nu - 1.0) \
            * torch.exp(- 2.0 * nu / l**2 * t) \
            * unnormalized_heat_kernel(dist_sq, torch.pow(2*t, 0.5))

    return result


class EuclideanIntegratedMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on the Euclidean manifold
    obtained by integrating over the heat kernel.

    Attributes
    ----------
    self.dim, dimension of the Euclidean space R^d on which the data handled by the kernel are living
    self.nu, smoothness parameter
    self.nb_points_integral, number of points used to compute the integral over the heat kernel

    Methods
    -------
    forward(point(s)1, point(s)2, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, nu=None, nu_prior=None, nb_points_integral=100, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the Euclidean space

        Optional parameters
        -------------------
        :param nu: smoothness parameter
        :param nu_prior: prior function on the smoothness parameter
        :param nb_points_integral: number of points used to compute the integral over the heat kernel
        """

        self.has_lengthscale = True
        super(EuclideanIntegratedMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Register smoothness parameter
        self.register_parameter(name="raw_nu", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))

        if nu_prior is not None:
            self.register_prior("nu_prior", nu_prior, lambda module: module.nu,
                                lambda module, value: module._set_nu(value))

        # A Positive constraint is defined on the smoothness parameter.
        self.register_constraint("raw_nu", Positive())

        # If the smoothness parameter is given, set it and deactivate its optimization by setting requires_grad false
        if nu is not None:
            self.nu = nu
            self.raw_nu.requires_grad = False

        # Dimension of the Euclidean data
        self.dim = dim

        # Number of points for the integral computation
        self.nb_points_integral = nb_points_integral

    @property
    def nu(self):
        return self.raw_nu_constraint.transform(self.raw_nu)

    @nu.setter
    def nu(self, value):
        self._set_nu(value)

    def _set_nu(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_nu)
        self.initialize(raw_nu=self.raw_nu_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Matérn kernel matrix between inputs x1 and x2 belonging to a Euclidean manifold by integrating over
        the Euclidean heat kernel.

        Parameters
        ----------
        :param x1: input points
        :param x2: input points

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        squared_distance = self.covar_dist(x1, x2, diag=diag, square_dist=True)

        # Evaluate integral
        shift = torch.log10(self.lengthscale).item()
        t_vals = torch.logspace(-3 + shift, 3 + shift,
                                self.nb_points_integral).to(device)
        integral_vals = torch.zeros([self.nb_points_integral] + list(squared_distance.shape)).to(device)
        for i in range(self.nb_points_integral):
            integral_vals[i] = link_function(squared_distance, t_vals[i], self.lengthscale, self.nu, self.dim)

        # Kernel
        kernel = torch.trapz(integral_vals, t_vals, dim=0)

        # Normalizing constant (exact value)
        # normalizating_cst = math.gamma(self.nu) / pow(2*self.nu/self.lengthscale**2, self.nu)

        # Evaluate the integral for the normalizing constant
        # This ensures better normalization as the imprecision of the integral is also reported on the value of the
        # normalizing constant
        integral_vals_normalizing_cst = torch.zeros(self.nb_points_integral).to(device)
        for i in range(self.nb_points_integral):
            integral_vals_normalizing_cst[i] = link_function(torch.zeros(1, 1).to(device), t_vals[i], self.lengthscale, self.nu,
                                                             self.dim)
        normalizating_cst = torch.trapz(integral_vals_normalizing_cst, t_vals, dim=0)

        return kernel/normalizating_cst
