"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu
"""

import os
import math
import numpy as np
import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive

from BoManifolds.Riemannian_utils.sphere_utils_torch import sphere_distance_torch
from BoManifolds.math_utils.gegenbauer_polynomials import gegenbauer_polynomial
from BoManifolds.math_utils.jacobi_theta_functions import jacobi_theta_function3

dirname = os.path.dirname(os.path.realpath(__file__))
device = torch.cuda.current_device()
torch.set_default_dtype(torch.float32)


class SphereRiemannianMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on the sphere manifold.

    Attributes
    ----------
    self.nu, smoothness parameter
    self.dim, dimension of the sphere S^d on which the data handled by the kernel are living
    self.serie_nb_terms, number of terms used to compute the summation formula of the kernel
    self.cst_nd, precomputed constant term of the summation formula of the kernel (term n, dimension d)
    self.zero_gpolynomials, precompute Gegenbauer polynomial for 0-distance (for the normalizing term of the kernel)

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, nu=None, serie_nb_terms=10, nu_prior=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the sphere S^d on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param nu: smoothness parameter, it will be selected automatically (optimized) if it is not given.
        :param serie_nb_terms: number of terms used to compute the summation formula of the kernel
        :param nu_prior: prior function on the smoothness parameter
        :param kwargs: additional arguments
        """

        self.has_lengthscale = True
        super(SphereRiemannianMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

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

        # Dimension of sphere data
        self.dim = dim

        # Number of term used to approximate the infinite serie approximating the kernel
        self.serie_nb_terms = serie_nb_terms

        # Precompute constant terms in the sum of the serie defining the kernel
        self.cst_nd = [compute_riemannian_matern_kernel_constant(n, self.dim).to(device) for n in range(self.serie_nb_terms)]

        # Precompute Gegenbauer polynomial for 0-distance (used to compute the normalizing term of the kernel)
        self.zero_gpolynomials = [gegenbauer_polynomial(n, (self.dim-1.)/2., torch.ones(1)).to(device) for n in range(self.serie_nb_terms)]

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
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute cos of distance
        cos_distance = torch.cos(sphere_distance_torch(x1, x2, diag=diag))

        # Compute serie and normalization factor
        kernel = torch.zeros_like(cos_distance)
        norm_factor = torch.zeros((1, 1)).to(device)
        exp_term0 = torch.pow(2*self.nu/self.lengthscale**2, -(self.nu + self.dim/2))
        for n in range(self.serie_nb_terms):
            # Compute exponential term normalized by exp_term0 to avoid too small values and numerical errors
            exp_term = torch.pow(2*self.nu/self.lengthscale**2 + n*(n+self.dim-1), -(self.nu + self.dim/2)) / exp_term0
            # Compute Gegenbauer polynomial
            gpolynomial = gegenbauer_polynomial(n, (self.dim - 1.) / 2., cos_distance)

            # Kernel serie's n-th term
            kernel += exp_term * self.cst_nd[n] * gpolynomial
            # Normalization factor serie's n-th term
            norm_factor += exp_term * self.cst_nd[n] * self.zero_gpolynomials[n]

        # Kernel
        return kernel / norm_factor


class SphereRiemannianGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the sphere manifold.

    Attributes
    ----------
    self.dim, dimension of the sphere S^d on which the data handled by the kernel are living
    self.serie_nb_terms, number of terms used to compute the summation formula of the kernel
    self.cst_nd, precomputed constant term of the summation formula of the kernel (term n, dimension d)
    self.zero_gpolynomials, precompute Gegenbauer polynomial for 0-distance (for the normalizing term of the kernel)

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, serie_nb_terms=10,  **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the sphere S^d on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param serie_nb_terms: number of terms used to compute the summation formula of the kernel
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SphereRiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Dimension of sphere data
        self.dim = dim

        # Number of term used to approximate the infinite serie approximating the kernel
        self.serie_nb_terms = serie_nb_terms

        # Precompute constant terms in the sum of the serie defining the kernel
        self.cst_nd = [compute_riemannian_matern_kernel_constant(n, self.dim).to(device) for n in range(self.serie_nb_terms)]

        # Precompute Gegenbauer polynomial for 0-distance (used to compute the normalizing term of the kernel)
        self.zero_gpolynomials = [gegenbauer_polynomial(n, (self.dim-1.)/2., torch.ones(1)).to(device) for n in range(self.serie_nb_terms)]

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute cos of distance
        cos_distance = torch.cos(sphere_distance_torch(x1, x2, diag=diag))

        # Compute serie and normalization factor
        kernel = torch.zeros_like(cos_distance)
        norm_factor = torch.zeros((1, 1)).to(device)
        for n in range(self.serie_nb_terms):
            # Compute exponential term
            exp_term = torch.exp(-self.lengthscale**2/2. * n * (n+self.dim-1))
            # Compute Gegenbauer polynomial
            gpolynomial = gegenbauer_polynomial(n, (self.dim-1.)/2., cos_distance)

            # Kernel serie's n-th term
            kernel += exp_term * self.cst_nd[n] * gpolynomial
            # Normalization factor serie's n-th term
            norm_factor += exp_term * self.cst_nd[n] * self.zero_gpolynomials[n]

        # Kernel
        return kernel / norm_factor


class SphereRiemannianIntegratedMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on the sphere manifold obtained
    by integrating over the heat kernel

    Attributes
    ----------
    self.nu, smoothness parameter
    self.dim, dimension of the sphere S^d on which the data handled by the kernel are living
    self.nb_points_integral, number of points used to compute the integral over the heat kernel
    self.serie_nb_terms, number of terms used to compute the summation formula of the kernel
    self.cst_nd, precomputed constant term of the summation formula of the kernel (term n, dimension d)
    self.zero_gpolynomials, precompute Gegenbauer polynomial for 0-distance (for the normalizing term of the kernel)

    Methods
    -------
    link_function(cos_distance, precomputed_Gegenbauer_polynomial, t)
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, nu=None, nu_prior=None, serie_nb_terms=10, nb_points_integral=100, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the sphere S^d on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param nu: smoothness parameter, it will be selected automatically (optimized) if it is not given.
        :param nu_prior: prior function on the smoothness parameter
        :param serie_nb_terms: number of terms used to compute the summation formula of the kernel
        :param nb_points_integral: number of points used to compute the integral over the heat kernel
        :param kwargs: additional arguments
        """

        self.has_lengthscale = True
        super(SphereRiemannianIntegratedMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Register smoothness parameter
        self.register_parameter(name="raw_nu", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))

        if nu_prior is not None:
            self.register_prior("nu_prior", nu_prior, lambda module: module.nu,
                                lambda module, value: module._set_nu(value))
            # self.register_prior("nu_prior", nu_prior, lambda: self.nu, lambda v: self._set_nu(v))

        # A Positive constraint is defined on the smoothness parameter.
        self.register_constraint("raw_nu", Positive())

        # If the smoothness parameter is given, set it and deactivate its optimization by setting requires_grad false
        if nu is not None:
            self.nu = nu
            self.raw_nu.requires_grad = False

        # Dimension of sphere data
        self.dim = dim

        # Number of points for the integral computation
        self.nb_points_integral = nb_points_integral

        # Number of term used to approximate the infinite serie approximating the kernel
        self.serie_nb_terms = serie_nb_terms

        # Precompute constant terms in the sum of the serie defining the kernel
        self.cst_nd = [compute_riemannian_matern_kernel_constant(n, self.dim).to(device) for n in range(self.serie_nb_terms)]

        # Precompute Gegenbauer polynomial for 0-distance (used to compute the normalizing term of the kernel)
        self.zero_gpolynomials = [gegenbauer_polynomial(n, (self.dim-1.)/2., torch.ones(1)).to(device) for n in range(self.serie_nb_terms)]

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

    def link_function(self, cos_distance, gpolynomial, t):
        """
        This function links the heat kernel to the Matérn kernel, i.e., the Matérn kernel correspond to the integral of
        this function from 0 to inf.

        Parameters
        ----------
        :param cos_distance: precomputed cosine distance between the inputs
        :param gpolynomial: precomputed Gegenbauer polynomials of the heat kernel
        :param t: heat kernel lengthscale

        Returns
        -------
        :return: link function between the heat and Matérn kernels

        """
        # Compute serie and normalization factor
        heat_kernel = torch.zeros_like(cos_distance).to(device)
        for n in range(self.serie_nb_terms):
            # Compute exponential term
            exp_term = torch.exp(- t * n * (n + self.dim - 1))
            # Kernel serie's n-th term
            heat_kernel += exp_term * self.cst_nd[n] * gpolynomial[n]

        result = torch.pow(t, self.nu + self.dim / 2 - 1.0) \
                 * torch.exp(- 2.0 * self.nu / self.lengthscale ** 2 * t) \
                 * heat_kernel

        return result

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the integrated Matérn kernel matrix between inputs x1 and x2 belonging to a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute cos of distance
        cos_distance = torch.cos(sphere_distance_torch(x1, x2, diag=diag))

        # Compute Gengenbauer polynomial for each serie's term
        gpolynomial = [gegenbauer_polynomial(n, (self.dim - 1.) / 2., cos_distance) for n in range(self.serie_nb_terms)]

        # Evaluate integral
        shift = torch.log10(self.lengthscale).item()
        t_vals = torch.logspace(-4 + shift, 3 + shift, self.nb_points_integral).to(device)
        integral_vals = torch.zeros([self.nb_points_integral] + list(cos_distance.shape)).to(device)
        for i in range(self.nb_points_integral):
            integral_vals[i] = self.link_function(cos_distance, gpolynomial, t_vals[i])
        # Kernel
        kernel = torch.trapz(integral_vals, t_vals, dim=0)

        # Evaluate the integral for the normalizing constant
        integral_vals_normalizing_cst = torch.zeros(self.nb_points_integral).to(device)
        for i in range(self.nb_points_integral):
            integral_vals_normalizing_cst[i] = self.link_function(torch.zeros(1, 1), self.zero_gpolynomials, t_vals[i])
        # Normalizing constant
        normalizating_cst = torch.trapz(integral_vals_normalizing_cst, t_vals, dim=0)

        # Kernel
        return kernel / normalizating_cst


def compute_riemannian_matern_kernel_constant(n, d):
    """
    This function computes the constant terms of the summation formula for the Matérn kernel on the sphere.

    Parameters
    ----------
    :param n: term n of the summation serie
    :param d: dimension of sphere S^d

    Returns
    -------
    :return: constant term c_nd

    """
    dn = (2*n+d-1) * math.gamma(n+d-1) / math.gamma(d) / math.gamma(n+1)
    # Compute Gegenbauer polynomial
    gpolynomial = gegenbauer_polynomial(n, (d-1.)/2., torch.ones(1))
    # Constant value
    cnd = dn * math.gamma((d-1.)/2.) / (2*math.pow(math.pi, (d-1.)/2.) * gpolynomial)
    return cnd


class CircleRiemannianGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the circle, i.e.,
    sphere manifold S¹.

    Attributes
    ----------
    self.serie_nb_terms, number of terms used to compute the Jacobi theta function of the kernel

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, serie_nb_terms=100,  **kwargs):
        """
        Initialisation.

        Parameters
        ----------

        Optional parameters
        -------------------
        :param serie_nb_terms: number of terms used to compute the summation formula of the kernel
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(CircleRiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Number of term used to compute the jacobi theta function
        self.serie_nb_terms = serie_nb_terms

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a circle / sphere manifold S^1.

        Parameters
        ----------
        :param x1: input points on the circle
        :param x2: input points on the circle

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        scaled_distance = sphere_distance_torch(x1, x2, diag=diag)/(2*np.pi)

        # Compute kernel equal to jacobi theta function
        q_param = torch.exp(-2 * np.pi**2 * self.lengthscale**2)
        kernel = jacobi_theta_function3(np.pi * scaled_distance, q_param).to(device)

        # Normalizing term
        norm_factor = jacobi_theta_function3(torch.zeros((1, 1)).to(device), q_param).to(device)

        # Kernel
        return kernel / norm_factor


class CircleRiemannianIntegratedMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on the circle manifold obtained
    by integrating over the heat kernel

    Attributes
    ----------
    self.nu, smoothness parameter
    self.dim, dimension of the sphere S^d on which the data handled by the kernel are living
    self.nb_points_integral, number of points used to compute the integral over the heat kernel
    self.serie_nb_terms, number of terms used to compute the summation formula of the kernel

    Methods
    -------
    link_function(distance, precomputed_Gegenbauer_polynomial, t)
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, nu=None, nu_prior=None, serie_nb_terms=20, nb_points_integral=50, **kwargs):
        """
        Initialisation.

        Parameters
        ----------

        Optional parameters
        -------------------
        :param nu: smoothness parameter, it will be selected automatically (optimized) if it is not given.
        :param nu_prior: prior function on the smoothness parameter
        :param serie_nb_terms: number of terms used to compute the summation formula of the kernel
        :param nb_points_integral: number of points used to compute the integral over the heat kernel
        :param kwargs: additional arguments
        """

        self.has_lengthscale = True
        super(CircleRiemannianIntegratedMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Register smoothness parameter
        self.register_parameter(name="raw_nu", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))

        if nu_prior is not None:
            self.register_prior("nu_prior", nu_prior, lambda module: module.nu,
                                lambda module, value: module._set_nu(value))

        # A Positive constraint is defined on the smoothness parameter.
        self.register_constraint("raw_nu", Positive())

        # Dimension of the sphere
        self.dim = 1

        # If the smoothness parameter is given, set it and deactivate its optimization by setting requires_grad false
        if nu is not None:
            self.nu = nu
            self.raw_nu.requires_grad = False

        # Number of points for the integral computation
        self.nb_points_integral = nb_points_integral

        # Number of term used to approximate the infinite serie approximating the kernel
        self.serie_nb_terms = serie_nb_terms

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

    def link_function(self, scaled_distance, t):
        """
        This function links the heat kernel to the Matérn kernel, i.e., the Matérn kernel correspond to the integral of
        this function from 0 to inf.

        Parameters
        ----------
        :param distance: precomputed scaled distance between the inputs
        :param t: heat kernel lengthscale

        Returns
        -------
        :return: link function between the heat and Matérn kernels

        """
        # Compute unnormalized heat kernel
        q_param = torch.exp(-4 * np.pi ** 2 * t)
        heat_kernel = jacobi_theta_function3(np.pi * scaled_distance, q_param).to(device)

        result = torch.pow(t, self.nu + self.dim / 2 - 1.0) \
                 * torch.exp(- 2.0 * self.nu / self.lengthscale ** 2 * t) \
                 * heat_kernel

        return result

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the integrated Matérn kernel matrix between inputs x1 and x2 belonging to a circle manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        scaled_distance = sphere_distance_torch(x1, x2, diag=diag) / (2 * np.pi)

        # Evaluate integral
        shift = torch.log10(self.lengthscale).item()
        t_vals = torch.logspace(-3 + shift, 1 + shift, self.nb_points_integral).to(device)
        integral_vals = torch.zeros([self.nb_points_integral] + list(scaled_distance.shape)).to(device)
        for i in range(self.nb_points_integral):
            integral_vals[i] = self.link_function(scaled_distance, t_vals[i])
        # Kernel
        kernel = torch.trapz(integral_vals, t_vals, dim=0)

        # Evaluate the integral for the normalizing constant
        integral_vals_normalizing_cst = torch.zeros(self.nb_points_integral).to(device)
        for i in range(self.nb_points_integral):
            integral_vals_normalizing_cst[i] = self.link_function(torch.zeros(1, 1).to(device), t_vals[i])
        # Normalizing constant
        normalizating_cst = torch.trapz(integral_vals_normalizing_cst, t_vals, dim=0)

        # Kernel
        return kernel / normalizating_cst


class SphereApproximatedGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the sphere manifold.
    This covariance matrix is an approximation of the SphereRiemannianGaussianKernel, where the Euclidean distance is
    replaced by the geodesic distance in a Euclidean-like RBF kernel.

    Attributes
    ----------
    self.dim, dimension of the sphere S^d on which the data handled by the kernel are living
    self.beta_min, minimum value of the inverse square lengthscale parameter beta

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, beta_min=None, beta_prior=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the sphere S^d on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param beta_min: minimum value of the inverse square lengthscale parameter beta.
                         If None, it is determined automatically.
        :param beta_prior: prior on the parameter beta
        :param kwargs: additional arguments
        """
        super(SphereApproximatedGaussianKernel, self).__init__(has_lengthscale=False, **kwargs)

        # Define beta_min
        if beta_min is None:
            if dim == 2:
                beta_min = 6.5
            elif dim == 3:
                beta_min = 2.
            elif dim == 4:
                beta_min = 1.2
            elif dim == 5:
                beta_min = 1.0
            elif dim <= 10:
                beta_min = 0.6
            elif dim <= 15:
                beta_min = 0.42
            elif dim <= 50:
                beta_min = 0.35
            elif 50 <= dim <= 160:
                beta_min = 0.21

        self.dim = dim
        self.beta_min = beta_min

        # Add beta parameter, corresponding to the inverse of the lengthscale parameter.
        beta_num_dims = 1
        self.register_parameter(name="raw_beta",
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, beta_num_dims)))

        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda: self.beta, lambda v: self._set_beta(v))

        # A GreaterThan constraint is defined on the lengthscale parameter to guarantee positive-definiteness.
        # The value of beta_min can be determined e.g. experimentally.
        self.register_constraint("raw_beta", GreaterThan(self.beta_min))

    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        self._set_beta(value)

    def _set_beta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        distance = sphere_distance_torch(x1, x2, diag=diag)
        distance2 = torch.mul(distance, distance)
        # Kernel
        exp_component = torch.exp(- distance2.mul(self.beta.double()))
        return exp_component


class SphereRiemannianLaplaceKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Laplace covariance matrix between input points on the sphere manifold.
    """
    def __init__(self, **kwargs):
        """
        Initialisation.

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SphereRiemannianLaplaceKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Laplace kernel matrix between inputs x1 and x2 belonging to a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        distance = sphere_distance_torch(x1, x2, diag=diag)
        # Kernel
        exp_component = torch.exp(- distance.div(torch.mul(self.lengthscale.double(), self.lengthscale.double())))
        return exp_component
