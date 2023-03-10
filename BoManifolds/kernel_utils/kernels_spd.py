"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu
"""

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive

from BoManifolds.Riemannian_utils.spd_utils_torch import logm_torch, vector_to_symmetric_matrix_mandel_torch, \
    affine_invariant_distance_torch, frobenius_distance_torch, exp_stein_divergence_torch, \
    spd_to_sphere_and_euclidean_torch, spd_to_so_and_euclidean_torch
from BoManifolds.kernel_utils.kernels_sphere import SphereRiemannianGaussianKernel, SphereRiemannianMaternKernel, \
    CircleRiemannianGaussianKernel, CircleRiemannianIntegratedMaternKernel
from BoManifolds.kernel_utils.kernels_so import SORiemannianMaternKernel, SORiemannianGaussianKernel
from BoManifolds.kernel_utils.kernels_euclidean import EuclideanIntegratedMaternKernel

device = torch.cuda.current_device()
torch.set_default_dtype(torch.float32)


class SpdRiemannianGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the SPD manifold.

    Attributes
    ----------
    self.dim, dimension of the SPD manifold S++^n on which the data handled by the kernel are living
    self.nb_points_integral, number of points used to compute the integral for the heat kernel

    Methods
    -------
    link_function(cos_distance, precomputed_Gegenbauer_polynomial, t)
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, nb_points_integral=50, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the SPD manifold S++^n on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param nb_points_integral: number of points used to compute the integral over the heat kernel
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SpdRiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Dimension of SPD manifold
        self.dim = dim
        if self.dim != 2:
            raise NotImplementedError

        # Number of points for the integral
        self.nb_points_integral = nb_points_integral

    def link_function(self, singular_values_diff, b_val):
        result = (2. * b_val + singular_values_diff) \
                 * torch.exp(-b_val * (b_val + singular_values_diff) / (self.lengthscale**2)) \
                 * torch.pow(torch.sinh(b_val) * torch.sinh(b_val + singular_values_diff), -1 / 2)
        return result

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a SPD manifold.

        Parameters
        ----------
        :param x1: input points on the SPD manifold
        :param x2: input points on the SPD manifold

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Transform input vector to matrix anc compute Cholesky decomposition
        x1 = torch.cholesky(vector_to_symmetric_matrix_mandel_torch(x1))
        x2 = torch.cholesky(vector_to_symmetric_matrix_mandel_torch(x2))

        # Expand dimensions to compute all matrix-matrix distances
        x1 = x1.unsqueeze(-3)
        x2 = x2.unsqueeze(-4)
        # Repeat x and y data along 1- and 0- dimensions to have ndata_x x ndata_y x dim x dim arrays
        x1 = torch.cat(x2.shape[-3] * [x1], dim=-3)
        x2 = torch.cat(x1.shape[-4] * [x2], dim=-4)

        # Compute singular values
        x1_inv_x2 = torch.bmm(x1.view(-1, self.dim, self.dim), torch.linalg.inv(x2).view(-1, self.dim, self.dim)).to(device)
        _, singular_values, _ = torch.linalg.svd(x1_inv_x2)

        # Reshape the singular values
        shape = list(x2.shape)[:-2]
        shape.append(singular_values.shape[-1])
        singular_values_rshp = singular_values.view(shape)

        # Compute difference and sum of squared singular values
        # Note: singular values that torch.linalg.svd outputs are sorted, the following code relies on this fact.
        H1, H2 = torch.log(singular_values_rshp[..., 0]), torch.log(singular_values_rshp[..., 1])
        # assert (H1 >= H2)
        r_H_sq = H1 ** 2 + H2 ** 2
        singular_values_diff = H1 - H2

        # Evaluate non-integral part
        nonintegral_part = torch.exp(-r_H_sq / (2 * self.lengthscale**2))

        # Evaluate integral part
        b_vals = torch.logspace(-5., 1., self.nb_points_integral).to(device)
        integral_vals = torch.zeros([self.nb_points_integral] + list(singular_values_diff.shape)).to(device)
        for i in range(self.nb_points_integral):
            integral_vals[i] = self.link_function(singular_values_diff, b_vals[i])
        # Kernel integral part
        kernel = torch.trapz(integral_vals, b_vals, dim=0) * nonintegral_part

        # Compute normalization factor
        # Evaluate only the integral part, as the non-integral part is equal to 1
        integral_vals_norm = torch.zeros([self.nb_points_integral]).to(device)
        for i in range(self.nb_points_integral):
            integral_vals_norm[i] = self.link_function(torch.zeros((1, 1), dtype=kernel.dtype).to(device), b_vals[i])
        # Kernel integral part
        norm_factor = torch.trapz(integral_vals_norm, b_vals, dim=0)

        # Kernel
        return kernel / norm_factor


class SpdRiemannianIntegratedMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on the sphere manifold obtained
    by integrating over the heat kernel

    Attributes
    ----------
    self.nu, smoothness parameter
    self.dim, dimension of the SPD manifold S++^n on which the data handled by the kernel are living
    self.nb_points_integral, number of points used to compute the integral over the heat kernel

    Methods
    -------
    link_function(cos_distance, precomputed_Gegenbauer_polynomial, t)
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):

    Static methods
    --------------
    """
    def __init__(self, dim, nu=None, nu_prior=None, nb_points_integral_b=50, nb_points_integral_t=50, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the SPD manifold S++^n on which the data handled by the kernel are living
        :param nu: smoothness parameter, it will be selected automatically (optimized) if it is not given.
        :param nu_prior: prior function on the smoothness parameter

        Optional parameters
        -------------------
        :param nb_points_integral: number of points used to compute the integral over the heat kernel
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SpdRiemannianIntegratedMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

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

        # Dimension of SPD manifold
        self.dim = dim
        if self.dim != 2:
            raise NotImplementedError

        # Number of points for the integral
        self.nb_points_integral_b = nb_points_integral_b
        self.nb_points_integral_t = nb_points_integral_t

    def link_function(self, singular_values_diff, r_H_sq, b_val, t):
        heat_kernel = (2. * b_val + singular_values_diff) \
                      * torch.exp(-b_val * (b_val + singular_values_diff) / (2 * t)) \
                      * torch.pow(torch.sinh(b_val) * torch.sinh(b_val + singular_values_diff), -1 / 2) \
                      * torch.exp(-r_H_sq / (4 * t))

        result = torch.pow(t, self.nu - 1.0) \
                 * torch.exp(- 2.0 * self.nu / self.lengthscale ** 2 * t) \
                 * heat_kernel
        
        return result

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a SPD manifold.

        Parameters
        ----------
        :param x1: input points on the SPD manifold
        :param x2: input points on the SPD manifold

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Transform input vector to matrix
        x1 = vector_to_symmetric_matrix_mandel_torch(x1)
        x2 = vector_to_symmetric_matrix_mandel_torch(x2)

        # Expand dimensions to compute all matrix-matrix distances
        x1 = x1.unsqueeze(-3)
        x2 = x2.unsqueeze(-4)
        # Repeat x and y data along 1- and 0- dimensions to have ndata_x x ndata_y x dim x dim arrays
        x1 = torch.cat(x2.shape[-3] * [x1], dim=-3)
        x2 = torch.cat(x1.shape[-4] * [x2], dim=-4)

        # Compute singular values
        x1_inv_x2 = torch.bmm(x1.view(-1, self.dim, self.dim), torch.linalg.inv(x2).view(-1, self.dim, self.dim))
        _, singular_values, _ = torch.linalg.svd(x1_inv_x2)
        singular_values = singular_values.to(device)

        # Reshape the singular values
        shape = list(x2.shape)[:-2]
        shape.append(singular_values.shape[-1])
        singular_values_rshp = singular_values.view(shape)

        # Compute difference and sum of squared singular values
        # Note: singular values that torch.linalg.svd outputs are sorted, the following code relies on this fact.
        H1, H2 = torch.log(singular_values_rshp[..., 0]), torch.log(singular_values_rshp[..., 1])
        # assert (H1 >= H2)
        r_H_sq = H1 ** 2 + H2 ** 2
        singular_values_diff = H1 - H2

        # # Evaluate non-integral part
        # nonintegral_part = torch.exp(-r_H_sq / (4 * t))
        #
        # Evaluate integral part
        shift = torch.log10(self.lengthscale).item()
        t_vals = torch.logspace(-2 + shift, 1.5 + shift, self.nb_points_integral_t).to(device)
        b_vals = torch.logspace(-5., 1., self.nb_points_integral_b).to(device)
        integral_vals = torch.zeros([self.nb_points_integral_t, self.nb_points_integral_b] +
                                    list(singular_values_diff.shape)).to(device)
        for i in range(self.nb_points_integral_t):
            for j in range(self.nb_points_integral_b):
                integral_vals[i, j] = self.link_function(singular_values_diff, r_H_sq, b_vals[j], t_vals[i])
        # Integral to obtain the heat kernel values
        heat_kernel_integral_vals = torch.trapz(integral_vals, b_vals, dim=1)
        # Integral over heat kernel to obtain the Matérn kernel values
        kernel = torch.trapz(heat_kernel_integral_vals, t_vals, dim=0)

        # Compute normalization factor
        # Evaluate only the integral part, as the non-integral part is equal to 1
        integral_vals_norm = torch.zeros([self.nb_points_integral_t, self.nb_points_integral_b]).to(device)
        for i in range(self.nb_points_integral_t):
            for j in range(self.nb_points_integral_b):
                integral_vals_norm[i, j] = self.link_function(torch.zeros((1, 1), dtype=kernel.dtype).to(device),
                                                              torch.zeros((1, 1), dtype=kernel.dtype).to(device),
                                                              b_vals[j], t_vals[i])
        # Integral to obtain the heat kernel values
        norm_factor_integral_vals = torch.trapz(integral_vals_norm, b_vals, dim=1)
        # Integral over heat kernel to obtain the Matérn kernel values
        norm_factor = torch.trapz(norm_factor_integral_vals, t_vals, dim=0)

        # Kernel
        return kernel / norm_factor


class SpdProductOfRSManifoldsRiemannianGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the SPD manifold by
    considering it as a product of the sphere and Euclidean manifold.
    To do so, SPD data are decomposed with the eigenvalue decomposition as S = QDQ', with Q in SO(n), D=diag(d), and d
    in R^n. Q is then transformed from SO to a sphere manifold.
    Notice that this class accepts only SPD manifold of dimension 2 and 3.

    Attributes
    ----------
    self.dim, dimension of the SPD manifold S++^n on which the data handled by the kernel are living
    self.dim_sphere, dimension of the sphere manifold composing the SPD manifold
    self.Euclidean_kernel, kernel handling the Euclidean manifold part of the SPD manifold
    self.sphere_kernel, kernel handling the sphere manifold part of the SPD manifold
    self.spd_kernel, product of the sphere and Euclidean kernels
    self.spd_input, True is inputs are given as Mandel form of SPD matrices,
                    False if given as concatenated Euclidean and Sphere vectors

    Methods
    -------
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):

    """
    def __init__(self, dim, spd_input=False, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the SPD manifold S++^n on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SpdProductOfRSManifoldsRiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None,
                                                                            **kwargs)

        # Dimension of SPD data
        self.dim = dim
        # Set the sphere dimension
        # If the dimension is not 2 or 3, raise an implementation error
        if self.dim == 2:
            self.dim_sphere = 1
        elif self.dim == 3:
            self.dim_sphere = 3
        else:
            raise NotImplementedError

        # Input given in Mandel form of SPD matrix or as concatenation of Euclidean and sphere vectors?
        self.spd_input = spd_input

        # Initialise the product of kernels
        self.Euclidean_kernel = gpytorch.kernels.RBFKernel(active_dims=torch.tensor(list(range(0, self.dim))))
        if self.dim_sphere == 1:
            self.sphere_kernel = CircleRiemannianGaussianKernel(active_dims=torch.tensor(list(range(self.dim,
                                                                                                    self.dim +
                                                                                                    self.dim_sphere
                                                                                                    +1))))
        else:
            self.sphere_kernel = SphereRiemannianGaussianKernel(dim=self.dim_sphere,
                                                                active_dims=torch.tensor(list(range(self.dim,
                                                                                                    self.dim +
                                                                                                    self.dim_sphere
                                                                                                    +1))))
        self.spd_kernel = self.Euclidean_kernel * self.sphere_kernel

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a SPD manifold by considering the SPD
        manifold as a product between a sphere and a Euclidean manifold. To do so, SPD data are decomposed with the
        eigenvalue decomposition as S = QDQ', with Q in SO(n) and D=diag(d), d in R^n. Q is then transformed from SO to
        a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the SPD manifold
        :param x2: input points on the SPD manifold

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Input given as SPD form
        if self.spd_input:
            # Transform input vector to matrix
            x1 = vector_to_symmetric_matrix_mandel_torch(x1)
            x2 = vector_to_symmetric_matrix_mandel_torch(x2)

            # Decompose SPD matrices in sphere (from SO(n)) and Euclidean parts using eigenvalue decompositions
            x_euclidean1, x_s1 = spd_to_sphere_and_euclidean_torch(x1)
            x_euclidean2, x_s2 = spd_to_sphere_and_euclidean_torch(x2)

            # Stack Euclidean and sphere data
            x1_joint = torch.cat((x_euclidean1, x_s1), dim=-1)
            x2_joint = torch.cat((x_euclidean2, x_s2), dim=-1)

        # Input given in the form of concatenated Euclidean and spherical parts
        else:
            x1_joint = x1
            x2_joint = x2

        # Kernel check
        # kernel_euclidean = gpytorch.kernels.RBFKernel()
        # kernel_sphere = SphereRiemannianGaussianKernel(dim=self.dim_sphere)
        # kernel_sphere = CircleRiemannianGaussianKernel()
        # kernel_euclidean.lengthscale = self.Euclidean_kernel.lengthscale
        # kernel_sphere.lengthscale = self.sphere_kernel.lengthscale
        # Ks = kernel_sphere.forward(x_s1, x_s2)
        # Ke = kernel_euclidean.forward(x_euclidean1, x_euclidean2)
        # Kspd = Ke * Ks

        # Kernel
        kernel = self.spd_kernel.forward(x1_joint, x2_joint)
        return kernel


class SpdProductOfRSManifoldsRiemannianMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on the SPD manifold by
    considering it as a product of the sphere and Euclidean manifold.
    To do so, SPD data are decomposed with the eigenvalue decomposition as S = QDQ', with Q in SO(n), D=diag(d), and d
    in R^n. Q is then transformed from SO to a sphere manifold.
    Notice that this class accepts only SPD manifold of dimension 2 and 3.

    Attributes
    ----------
    self.dim, dimension of the SPD manifold S++^n on which the data handled by the kernel are living
    self.dim_sphere, dimension of the sphere manifold composing the SPD manifold
    self.Euclidean_kernel, kernel handling the Euclidean manifold part of the SPD manifold
    self.sphere_kernel, kernel handling the sphere manifold part of the SPD manifold
    self.spd_kernel, product of the sphere and Euclidean kernels
    self.spd_input, True is inputs are given as Mandel form of SPD matrices,
                    False if given as concatenated Euclidean and Sphere vectors

    Methods
    -------
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):
    """
    def __init__(self, dim, nu=None, nu_sphere=None, nu_prior_sphere=None, nu_euclidean=None, nu_prior_euclidean=None,
                 spd_input=False, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the SPD manifold S++^n on which the data handled by the kernel are living
        :param nu_sphere: shared smoothness parameter for both kernels, it will be selected automatically (optimized)
                          if it is not given.
        :param nu_sphere: smoothness parameter for the sphere kernel, it will be selected automatically (optimized)
                          if it is not given.
        :param nu_prior_sphere: prior function on the smoothness parameter for the sphere kernel
        :param nu_euclidean: smoothness parameter for the Euclidean kernel, it will be selected automatically
                             (optimized) if it is not given.
        :param nu_prior_euclidean: prior function on the smoothness parameter for the Euclidean kernel

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SpdProductOfRSManifoldsRiemannianMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None,
                                                                          **kwargs)

        # Dimension of SPD data
        self.dim = dim
        # Set the sphere dimension
        # If the dimension is not 2 or 3, raise an implementation error
        if self.dim == 2:
            self.dim_sphere = 1
        elif self.dim == 3:
            self.dim_sphere = 3
        else:
            raise NotImplementedError

        # If a common smoothness parameter is given, redefine the smoothness of the kernels
        if nu:
            nu_euclidean = nu
            nu_sphere = nu

        # Input given in Mandel form of SPD matrix or as concatenation of Euclidean and sphere vectors?
        self.spd_input = spd_input

        # Initialise the product of kernels
        self.Euclidean_kernel = EuclideanIntegratedMaternKernel(self.dim, nu=nu_euclidean, nu_prior=nu_prior_euclidean,
                                                                active_dims=torch.tensor(list(range(0, self.dim))))
        if self.dim_sphere == 1:
            self.sphere_kernel = CircleRiemannianIntegratedMaternKernel(nu=nu_sphere, nu_prior=nu_prior_sphere,
                                                                        active_dims=
                                                                        torch.tensor(list(range(self.dim, self.dim +
                                                                                                self.dim_sphere +1))))
        else:
            self.sphere_kernel = SphereRiemannianMaternKernel(dim=self.dim_sphere, nu=nu_sphere,
                                                              nu_prior=nu_prior_sphere,
                                                              active_dims=torch.tensor(list(range(self.dim,
                                                                                                  self.dim +
                                                                                                  self.dim_sphere+1))))
        self.spd_kernel = self.Euclidean_kernel * self.sphere_kernel

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a SPD manifold by considering the SPD
        manifold as a product between a sphere and a Euclidean manifold. To do so, SPD data are decomposed with the
        eigenvalue decomposition as S = QDQ', with Q in SO(n) and D=diag(d), d in R^n. Q is then transformed from SO to
        a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the SPD manifold
        :param x2: input points on the SPD manifold

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Input given as SPD form
        if self.spd_input:
            # Transform input vector to matrix
            x1 = vector_to_symmetric_matrix_mandel_torch(x1)
            x2 = vector_to_symmetric_matrix_mandel_torch(x2)

            # Decompose SPD matrices in sphere (from SO(n)) and Euclidean parts using eigenvalue decompositions
            x_euclidean1, x_s1 = spd_to_sphere_and_euclidean_torch(x1)
            x_euclidean2, x_s2 = spd_to_sphere_and_euclidean_torch(x2)

            # Stack Euclidean and sphere data
            x1_joint = torch.cat((x_euclidean1, x_s1), dim=-1)
            x2_joint = torch.cat((x_euclidean2, x_s2), dim=-1)

        # Input given in the form of concatenated Euclidean and spherical parts
        else:
            x1_joint = x1
            x2_joint = x2

        # Kernel check
        # kernel_euclidean = EuclideanIntegratedMaternKernel(self.dim, nu=self.Euclidean_kernel.nu)
        # # kernel_sphere = SphereRiemannianMaternKernel(dim=self.dim_sphere)
        # kernel_sphere = CircleRiemannianIntegratedMaternKernel(nu=self.sphere_kernel.nu)
        # kernel_euclidean.lengthscale = self.Euclidean_kernel.lengthscale
        # kernel_sphere.lengthscale = self.sphere_kernel.lengthscale
        # Ks = kernel_sphere.forward(x_s1, x_s2)
        # Ke = kernel_euclidean.forward(x_euclidean1, x_euclidean2)
        # Kspd = Ke * Ks

        # Kernel
        kernel = self.spd_kernel.forward(x1_joint, x2_joint)
        return kernel


class SpdProductOfRSOManifoldsRiemannianGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the SPD manifold by
    considering it as a product of the SO and Euclidean manifold.
    To do so, SPD data are decomposed with the eigenvalue decomposition as S = QDQ', with Q in SO(n), D=diag(d), and d
    in R^n.
    Notice that this class accepts only SPD manifold until dimension 6.

    Attributes
    ----------
    self.dim, dimension of the SPD manifold S++^n on which the data handled by the kernel are living
    self.dim_sphere, dimension of the sphere manifold composing the SPD manifold
    self.Euclidean_kernel, kernel handling the Euclidean manifold part of the SPD manifold
    self.so_kernel, kernel handling the SO manifold part of the SPD manifold
    self.spd_kernel, product of the sphere and Euclidean kernels
    self.spd_input, True is inputs are given as Mandel form of SPD matrices,
                    False if given as concatenated Euclidean and Sphere vectors

    Methods
    -------
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):

    """
    def __init__(self, dim, spd_input=False, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the SPD manifold S++^n on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SpdProductOfRSOManifoldsRiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None,
                                                                               **kwargs)

        # Dimension of SPD data
        self.dim = dim

        if self.dim > 6:
            raise NotImplementedError

        # Input given in Mandel form of SPD matrix or as concatenation of Euclidean and sphere vectors?
        self.spd_input = spd_input

        # Initialise the product of kernels
        self.Euclidean_kernel = gpytorch.kernels.RBFKernel(active_dims=torch.tensor(list(range(0, self.dim))))
        self.so_kernel = SORiemannianGaussianKernel(dim=self.dim,
                                                    active_dims=torch.tensor(list(range(self.dim,
                                                                                        self.dim + self.dim**2))))
        self.spd_kernel = self.Euclidean_kernel * self.so_kernel

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a SPD manifold by considering the SPD
        manifold as a product between a sphere and a Euclidean manifold. To do so, SPD data are decomposed with the
        eigenvalue decomposition as S = QDQ', with Q in SO(n) and D=diag(d), d in R^n. Q is then transformed from SO to
        a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the SPD manifold
        :param x2: input points on the SPD manifold

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Input given as SPD form
        if self.spd_input:
            # Transform input vector to matrix
            x1 = vector_to_symmetric_matrix_mandel_torch(x1)
            x2 = vector_to_symmetric_matrix_mandel_torch(x2)

            # Decompose SPD matrices in SO and Euclidean parts using eigenvalue decompositions
            x_euclidean1, x_so1 = spd_to_so_and_euclidean_torch(x1)
            x_euclidean2, x_so2 = spd_to_so_and_euclidean_torch(x2)

            # Flatten SO data
            x_so1_shape = list(x_so1.shape[:-2])
            x_so1_shape.append(int(x_so1.shape[-1]**2))
            x_so2_shape = list(x_so2.shape[:-2])
            x_so2_shape.append(int(x_so2.shape[-1] ** 2))

            # Stack Euclidean and sphere data
            x1_joint = torch.cat((x_euclidean1, x_so1.reshape(x_so1_shape)), dim=-1)
            x2_joint = torch.cat((x_euclidean2, x_so2.reshape(x_so2_shape)), dim=-1)

        # Input given in the form of concatenated Euclidean and so parts
        else:
            x1_joint = x1
            x2_joint = x2

        # Kernel
        kernel = self.spd_kernel.forward(x1_joint, x2_joint)
        return kernel


class SpdProductOfRSOManifoldsRiemannianMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on the SPD manifold by
    considering it as a product of the SO and Euclidean manifold.
    To do so, SPD data are decomposed with the eigenvalue decomposition as S = QDQ', with Q in SO(n), D=diag(d), and d
    in R^n.
    Notice that this class accepts only SPD manifold of dimension until 6.

    Attributes
    ----------
    self.dim, dimension of the SPD manifold S++^n on which the data handled by the kernel are living
    self.dim_sphere, dimension of the sphere manifold composing the SPD manifold
    self.Euclidean_kernel, kernel handling the Euclidean manifold part of the SPD manifold
    self.so_kernel, kernel handling the SO manifold part of the SPD manifold
    self.spd_kernel, product of the SO and Euclidean kernels
    self.spd_input, True is inputs are given as Mandel form of SPD matrices,
                    False if given as concatenated Euclidean and Sphere vectors

    Methods
    -------
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):
    """
    def __init__(self, dim, nu=None, nu_so=None, nu_prior_so=None, nu_euclidean=None, nu_prior_euclidean=None,
                 spd_input=False, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the SPD manifold S++^n on which the data handled by the kernel are living
        :param nu_so: shared smoothness parameter for both kernels, it will be selected automatically (optimized)
                      if it is not given.
        :param nu_so: smoothness parameter for the sphere kernel, it will be selected automatically (optimized)
                      if it is not given.
        :param nu_prior_sphere: prior function on the smoothness parameter for the sphere kernel
        :param nu_euclidean: smoothness parameter for the Euclidean kernel, it will be selected automatically
                             (optimized) if it is not given.
        :param nu_prior_euclidean: prior function on the smoothness parameter for the Euclidean kernel

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SpdProductOfRSOManifoldsRiemannianMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None,
                                                                             **kwargs)

        # Dimension of SPD data
        self.dim = dim
        if self.dim > 6:
            raise NotImplementedError

        # If a common smoothness parameter is given, redefine the smoothness of the kernels
        if nu:
            nu_euclidean = nu
            nu_so = nu

        # Input given in Mandel form of SPD matrix or as concatenation of Euclidean and sphere vectors?
        self.spd_input = spd_input

        # Initialise the product of kernels
        self.Euclidean_kernel = EuclideanIntegratedMaternKernel(self.dim, nu=nu_euclidean, nu_prior=nu_prior_euclidean,
                                                                active_dims=torch.tensor(list(range(0, self.dim))))
        self.so_kernel = SORiemannianMaternKernel(dim=self.dim, nu=nu_so, nu_prior=nu_prior_so,
                                                  active_dims=torch.tensor(list(range(self.dim,
                                                                                      self.dim + self.dim**2))))
        self.spd_kernel = self.Euclidean_kernel * self.so_kernel

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a SPD manifold by considering the SPD
        manifold as a product between a sphere and a Euclidean manifold. To do so, SPD data are decomposed with the
        eigenvalue decomposition as S = QDQ', with Q in SO(n) and D=diag(d), d in R^n. Q is then transformed from SO to
        a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the SPD manifold
        :param x2: input points on the SPD manifold

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Input given as SPD form
        if self.spd_input:
            # Transform input vector to matrix
            x1 = vector_to_symmetric_matrix_mandel_torch(x1)
            x2 = vector_to_symmetric_matrix_mandel_torch(x2)

            # Decompose SPD matrices in SO and Euclidean parts using eigenvalue decompositions
            x_euclidean1, x_so1 = spd_to_so_and_euclidean_torch(x1)
            x_euclidean2, x_so2 = spd_to_so_and_euclidean_torch(x2)

            # Flatten SO data
            x_so1_shape = list(x_so1.shape[:-2])
            x_so1_shape.append(int(x_so1.shape[-1] ** 2))
            x_so2_shape = list(x_so2.shape[:-2])
            x_so2_shape.append(int(x_so2.shape[-1] ** 2))

            # Stack Euclidean and sphere data
            x1_joint = torch.cat((x_euclidean1, x_so1.reshape(x_so1_shape)), dim=-1)
            x2_joint = torch.cat((x_euclidean2, x_so2.reshape(x_so2_shape)), dim=-1)

        # Input given in the form of concatenated Euclidean and spherical parts
        else:
            x1_joint = x1
            x2_joint = x2

        # Kernel
        kernel = self.spd_kernel.forward(x1_joint, x2_joint)
        return kernel


class SpdAffineInvariantApproximatedGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the SPD manifold using
    the affine-invariant distance.

    Attributes
    ----------
    self.dim: dimension of the SPD on which the data handled by the kernel are living
    self.beta_min: minimum value of the inverse square lengthscale parameter beta

    Methods
    -------
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):

    Static methods
    --------------
    """
    def __init__(self, dim, beta_min=None, beta_prior=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the SPD on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param beta_min: minimum value of the inverse square lengthscale parameter beta.
                         If None, it is determined automatically.
        :param beta_prior: prior on the parameter beta
        :param kwargs: additional arguments
        """
        super(SpdAffineInvariantApproximatedGaussianKernel, self).__init__(has_lengthscale=False, **kwargs)

        if beta_min is None:
            # Define beta_min
            if dim == 2:
                beta_min = 0.6
            elif dim == 3:
                beta_min = 0.5
            elif dim == 5:
                beta_min = 0.25
            elif dim == 7:
                beta_min = 0.22
            elif dim == 10:
                beta_min = 0.2
            elif dim == 12:
                beta_min = 0.16

        self.dim = dim
        self.beta_min = beta_min

        # Add beta parameter, corresponding to the inverse of the lengthscale parameter.
        beta_num_dims = 1
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, beta_num_dims)))

        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda: self.beta, lambda v: self._set_beta(v))

        # A GreaterThan constraint is defined on the lengthscale parameter to guarantee the positive-definiteness of the kernel.
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

    def forward(self, x1, x2, diagonal_distance=False, **params):
        """
        Compute the Gaussian kernel matrix between inputs x1 and x2 belonging to a SPD manifold.

        Parameters
        ----------
        :param x1: input points on the SPD manifold
        :param x2: input points on the SPD manifold

        Optional parameters
        -------------------
        :param diagonal_distance: Whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Transform input vector to matrix
        x1 = vector_to_symmetric_matrix_mandel_torch(x1)
        x2 = vector_to_symmetric_matrix_mandel_torch(x2)

        # Compute distance
        distance = affine_invariant_distance_torch(x1, x2, diagonal_distance=diagonal_distance)
        distance2 = torch.mul(distance, distance)

        exp_component = torch.exp(- distance2.mul(self.beta))

        return exp_component


class SpdLogEuclideanGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the SPD manifold using
    the log-Euclidean distance.

    Methods
    -------
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):

    Static methods
    --------------
    """
    def __init__(self, **kwargs):
        """
        Initialisation.

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SpdLogEuclideanGaussianKernel, self).__init__(ard_num_dims=None, **kwargs)

    def forward(self, x1, x2, diagonal_distance=False, **params):
        """
        Compute the Gaussian kernel matrix between inputs x1 and x2 belonging to a SPD manifold.

        Parameters
        -------------------
        :param x1: input points on the SPD manifold
        :param x2: input points on the SPD manifold

        Optional parameters
        -------------------
        :param diag: Whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------------------
        :return: kernel matrix between x1 and x2
        """
        # Transform input vector to matrix
        x1 = vector_to_symmetric_matrix_mandel_torch(x1)
        x2 = vector_to_symmetric_matrix_mandel_torch(x2)

        # Compute the log of the matrices
        init_shape = list(x1.shape)
        dim = x1.shape[-1]
        x1 = x1.view(-1, dim, dim)
        nb_data = x1.shape[0]
        log_x1 = torch.zeros_like(x1)
        for n in range(nb_data):
            log_x1[n] = logm_torch(x1[n])
        log_x1 = log_x1.view(init_shape)

        init_shape = list(x2.shape)
        x2 = x2.view(-1, dim, dim)
        nb_data = x2.shape[0]
        log_x2 = torch.zeros_like(x2)
        for n in range(nb_data):
            log_x2[n] = logm_torch(x2[n])
        log_x2 = log_x2.view(init_shape)

        # Compute distance
        distance = frobenius_distance_torch(log_x1, log_x2, diagonal_distance=diagonal_distance)
        distance2 = torch.mul(distance, distance)

        exp_component = torch.exp(- distance2.div(torch.mul(self.lengthscale.double(), self.lengthscale.double())))

        return exp_component
