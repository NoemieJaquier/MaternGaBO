"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu
"""

import torch
import gpytorch
from gpytorch.constraints import Positive

from BoManifolds.Riemannian_utils.hyperbolic_utils_torch import lorentz_distance_torch

device = torch.cuda.current_device()
torch.set_default_dtype(torch.float32)


class HyperbolicRiemannianMillsonGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the hyperbolic manifold
    using Millson's formula.

    Attributes
    ----------
    self.dim, dimension of the hyperbolic H^n on which the data handled by the kernel are living
    self.nb_points_integral, number of points used to compute the integral for the heat kernel

    Methods
    -------
    link_function(cosh_distance, b)
    forward(point1_in_hyperbolic, point2_in_hyperbolic, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, nb_points_integral=1000, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the hyperbolic H^n on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param nb_points_integral: number of points used to compute the integral for the heat kernel
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(HyperbolicRiemannianMillsonGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Dimension of hyperbolic manifold (data dimension = self.dim + 1
        self.dim = dim

        # Number of points for the integral computation
        self.nb_points_integral = nb_points_integral

    def forward_dim1(self, distance):
        # Kernel
        kernel = torch.exp(- torch.pow(distance, 2) / (2 * self.lengthscale ** 2))

        return kernel

    def forward_dim2(self, distance):
        # Evaluate integral part
        s_vals = torch.linspace(1e-2, 100., self.nb_points_integral).to(device) + distance.unsqueeze(-1)
        integral_vals = s_vals * torch.exp(-s_vals**2/(2*self.lengthscale**2)) \
                        / torch.sqrt(torch.cosh(s_vals) - torch.cosh(distance.unsqueeze(-1)))

        # Kernel integral part
        kernel = torch.trapz(integral_vals, s_vals, dim=-1)

        # Normalizing constant
        s_vals_normalizing = torch.linspace(1e-2, 100., self.nb_points_integral).to(device)
        integral_vals_normalizing = s_vals_normalizing * \
                                    torch.exp(-s_vals_normalizing ** 2 / (2 * self.lengthscale ** 2)) \
                                    / torch.sqrt(torch.cosh(s_vals_normalizing) - 1.)
        normalizing_cst = torch.trapz(integral_vals_normalizing, s_vals_normalizing, dim=-1)

        # return kernel
        return kernel / normalizing_cst

    def forward_dim3(self, distance):
        # Kernel (simplifying non-distance-related terms)
        kernel = torch.exp(- torch.pow(distance, 2) / (2 * self.lengthscale ** 2))
        # kernel = kernel * distance / torch.sinh(distance)  # Adding 1e-8 avoids numerical issues around d=0
        kernel = kernel * (distance+1e-8) / torch.sinh(distance+1e-8)  # Adding 1e-8 avoids numerical issues around d=0
        return kernel

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a hyperbolic manifold.

        Parameters
        ----------
        :param x1: input points on the hyperbolic manifold
        :param x2: input points on the hyperbolic manifold

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute hyperbolic distance
        distance = lorentz_distance_torch(x1, x2, diag=diag)

        # If dimension is 1, 2, or 3, compute the kernel directly
        if self.dim == 1:
            kernel = self.forward_dim1(distance)
        elif self.dim == 2:
            kernel = self.forward_dim2(distance)
        elif self.dim == 3:
            kernel = self.forward_dim3(distance)
        else:
            if self.dim % 2 == 0:  # even dimension > 2
                # Number of necessary derivatives
                nb_derivatives = int((self.dim - 2) / 2)
                # Approximate distance smaller than a constant
                min_distance = 1e-6
                distance[distance < min_distance] = min_distance
                distance.requires_grad = True
                # Compute sinh of hyperbolic distance
                sinhdistance = torch.sinh(distance)

                # Compute derivative(s)
                derivative_part = self.forward_dim2(distance)
                for i in range(nb_derivatives):
                    gradient = torch.autograd.grad(derivative_part, distance,
                                                   grad_outputs=torch.ones_like(distance), create_graph=True)[0]
                    derivative_part = gradient / sinhdistance

                # Normalizing term
                normalizing_distance = min_distance * torch.ones((1, 1), dtype=distance.dtype, requires_grad=True)
                derivative_part_normalizing = self.forward_dim2(normalizing_distance)
                for i in range(nb_derivatives):
                    gradient_normalizing = torch.autograd.grad(derivative_part_normalizing,
                                                               normalizing_distance,
                                                               grad_outputs=torch.ones_like(normalizing_distance),
                                                               create_graph=True)[0]
                    derivative_part_normalizing = gradient_normalizing / torch.sinh(normalizing_distance)

                kernel = -derivative_part #/ torch.pow(sinhdistance, nb_derivatives)
                normalizing_cst = -derivative_part_normalizing #/ torch.pow(torch.sinh(normalizing_distance), nb_derivatives)
                kernel = kernel / normalizing_cst

            else:  # odd dimension > 3
                # Number of necessary derivatives
                nb_derivatives = int((self.dim - 3) / 2)
                # Approximate distance smaller than a constant
                min_distance = 1e-6
                distance[distance < min_distance] = min_distance
                distance.requires_grad = True
                # Compute sinh of hyperbolic distance
                sinhdistance = torch.sinh(distance)

                # Compute derivative(s)
                derivative_part = self.forward_dim3(distance)
                for i in range(nb_derivatives):
                    gradient = torch.autograd.grad(derivative_part, distance,
                                                   grad_outputs=torch.ones_like(distance), create_graph=True)[0]
                    derivative_part = gradient / sinhdistance

                # Normalizing term
                normalizing_distance = min_distance * torch.ones((1, 1), dtype=distance.dtype, requires_grad=True)
                derivative_part_normalizing = self.forward_dim3(normalizing_distance)
                for i in range(nb_derivatives):
                    gradient_normalizing = torch.autograd.grad(derivative_part_normalizing,
                                                                      normalizing_distance,
                                                                      grad_outputs=torch.ones_like(normalizing_distance),
                                                                      create_graph=True)[0]
                    derivative_part_normalizing = gradient_normalizing / torch.sinh(normalizing_distance)

                kernel = -derivative_part #/ torch.pow(sinhdistance, nb_derivatives)
                normalizing_cst = -derivative_part_normalizing #/ torch.pow(torch.sinh(normalizing_distance), nb_derivatives)

                kernel = kernel / normalizing_cst

        # Kernel
        return kernel


class HyperbolicRiemannianMillsonIntegratedMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on the hyperbolic manifold
    obtained by integrating over the heat kernel using Millson's formula.

    Attributes
    ----------
    self.nu, smoothness parameter
    self.dim, dimension of the hyperbolic H^n on which the data handled by the kernel are living
    self.nb_points_integral, number of points used to compute the integral over the heat kernel

    Methods
    -------
    link_function(cosh_distance, b)
    forward(point1_in_hyperbolic, point2_in_hyperbolic, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, nu=None, nu_prior=None, nb_points_integral=50, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the hyperbolic H^n on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param nu: smoothness parameter, it will be selected automatically (optimized) if it is not given.
        :param nu_prior: prior function on the smoothness parameter
        :param nb_points_integral: number of points used to compute the integral over the heat kernel
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(HyperbolicRiemannianMillsonIntegratedMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None,
                                                                                **kwargs)
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

        # Dimension of hyperbolic manifold (data dimension = self.dim + 1
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

    def link_function_dim1(self, distance, t):
        """
        This function links the heat kernel to the Matérn kernel, i.e., the Matérn kernel correspond to the integral of
        this function from 0 to inf.

        Parameters
        ----------
        :param distance: precomputed distance between the inputs
        :param t: heat kernel lengthscale

        Returns
        -------
        :return: link function between the heat and Matérn kernels

        """
        heat_kernel = torch.exp(- torch.pow(distance, 2) / (4 * t))

        result = torch.pow(t, self.nu - 1.0) \
                 * torch.exp(- 2.0 * self.nu / self.lengthscale ** 2 * t) \
                 * heat_kernel

        return result

    def link_function_dim2(self, distance, t):
        """
        This function links the heat kernel to the Matérn kernel, i.e., the Matérn kernel correspond to the integral of
        this function from 0 to inf.

        Parameters
        ----------
        :param distance: precomputed distance between the inputs
        :param t: heat kernel lengthscale
        :param b: heat kernel integral parameter

        Returns
        -------
        :return: link function between the heat and Matérn kernels

        """
        # TODO the behavior of this kernel is not so stable around zero distance due to the division in the computation
        #  of the integral value and depends on the start of the s_vals interval
        s_vals = torch.linspace(1.5e-1, 100., self.nb_points_integral).to(device) + distance.unsqueeze(-1)
        integral_vals = s_vals * torch.exp(-s_vals ** 2 / (4 * t)) \
                        / torch.sqrt(torch.cosh(s_vals) - torch.cosh(distance.unsqueeze(-1)))

        # Kernel integral part
        heat_kernel = torch.trapz(integral_vals, s_vals, dim=-1)

        result = torch.pow(t, self.nu - 1.0) \
                 * torch.exp(- 2.0 * self.nu / self.lengthscale ** 2 * t) \
                 * heat_kernel

        return result

    def link_function_dim3(self, distance, t):
        """
        This function links the heat kernel to the Matérn kernel, i.e., the Matérn kernel correspond to the integral of
        this function from 0 to inf.

        Parameters
        ----------
        :param distance: precomputed distance between the inputs
        :param t: heat kernel lengthscale

        Returns
        -------
        :return: link function between the heat and Matérn kernels

        """
        heat_kernel = torch.exp(- torch.pow(distance, 2) / (4 * t))
        heat_kernel = heat_kernel * (distance + 1e-8) / torch.sinh(distance + 1e-8)  # Adding 1e-8 avoids numerical issues around d=0

        result = torch.pow(t, self.nu - 1.0) \
                 * torch.exp(- 2.0 * self.nu / self.lengthscale ** 2 * t) \
                 * heat_kernel

        return result

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Matérn kernel matrix between inputs x1 and x2 belonging to a hyperbolic manifold.

        Parameters
        ----------
        :param x1: input points on the hyperbolic manifold
        :param x2: input points on the hyperbolic manifold

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute hyperbolic distance
        distance = lorentz_distance_torch(x1, x2, diag=diag).to(device)

        # If dimension is 1 or 3, compute the kernel directly
        if self.dim == 1 or self.dim == 3 or self.dim == 2:
            if self.dim == 1:
                link_function = self.link_function_dim1
            elif self.dim == 2:
                link_function = self.link_function_dim2
            else:
                link_function = self.link_function_dim3

            # Evaluate integral
            shift = torch.log10(self.lengthscale).item()
            t_vals = torch.logspace(-2.5 + shift, 1.5 + shift,
                                    self.nb_points_integral).to(device)
            integral_vals = torch.zeros([self.nb_points_integral] + list(distance.shape)).to(device)
            for i in range(self.nb_points_integral):
                integral_vals[i] = link_function(distance, t_vals[i])

            # Kernel
            kernel = torch.trapz(integral_vals, t_vals, dim=0)

            # Evaluate the integral for the normalizing constant
            integral_vals_normalizing_cst = torch.zeros(self.nb_points_integral).to(device)
            for i in range(self.nb_points_integral):
                integral_vals_normalizing_cst[i] = link_function(torch.zeros(1, 1).to(device), t_vals[i])
            normalizating_cst = torch.trapz(integral_vals_normalizing_cst, t_vals, dim=0)

            kernel = kernel / normalizating_cst

        else:
            raise NotImplementedError

        # Kernel
        return kernel

