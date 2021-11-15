"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier and Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
"""

import numpy as np
import torch

from pymanopt.manifolds import *
from BoManifolds.pymanopt_addons.manifolds import *

from BoManifolds.Riemannian_utils.sphere_utils import unit_sphere_to_rotation_matrix
from BoManifolds.Riemannian_utils.spd_utils import vector_to_symmetric_matrix_mandel, symmetric_matrix_to_vector_mandel


def get_base(manifold):
    if isinstance(manifold, Euclidean):
        return np.zeros((1, manifold._shape[0]))

    elif isinstance(manifold, Sphere):
        # Dimension of the manifold
        dimension = manifold._shape[0]

        # The base is fixed at (1, 0, 0, ...) for simplicity. Therefore, the tangent plane is aligned with the axis x.
        # The first coordinate of x_proj is always 0, so that vectors in the tangent space can be expressed in a dim-1
        # dimensional space by simply ignoring the first coordinate.
        base = np.zeros((1, dimension))
        base[0, 0] = 1.
        return base

    elif isinstance(manifold, SymmetricPositiveDefinite):
        return symmetric_matrix_to_vector_mandel(2 * np.eye(manifold._n))[None]

    elif isinstance(manifold, PositiveDefiniteProductEuclideanSphere):
        eucl_base = 2. * np.ones(manifold.manifolds[0].dim)
        sphere_base = np.zeros(manifold.manifolds[1]._shape[0])
        sphere_base[0] = 1.
        return np.hstack((eucl_base, sphere_base))[None]

    elif isinstance(manifold, HyperbolicLorentz):
        # Dimension of the manifold
        dimension = manifold._shape[0] + 1

        # The base is fixed at (1, 0, 0, ...) for simplicity.
        base = np.zeros((1, dimension))
        base[0, 0] = 1.
        return base

    elif isinstance(manifold, Torus):
        dimension = 2*len(manifold.manifolds)
        base = np.zeros((1, dimension))
        base[:, 0::2] = 1.
        return base

    elif isinstance(manifold, SpecialOrthogonalGroup):
        dimension = manifold._n
        base = np.eye(dimension)
        return base.reshape((dimension**2))[None]

    else:
        raise RuntimeError("The base is not implemented for this manifold.")


def preprocess_manifold_data(manifold, x, cholesky=False):
    base = get_base(manifold)

    if isinstance(manifold, Euclidean):
        return x

    elif isinstance(manifold, Sphere):
        x_proj = manifold.log(base, x)

        # Remove first dim
        return x_proj[:, 1:]

    elif isinstance(manifold, SymmetricPositiveDefinite):
        # Dimension
        dimension = manifold._n

        if not cholesky:
            # To symmetric matrix (Mandel notation)
            x = vector_to_symmetric_matrix_mandel(x[0])

        else:
            x = x[0]

            # Verify that Cholesky decomposition does not have zero
            if x.size - np.count_nonzero(x):
                x += 1e-6
            # Add also a small value to too-close-to-zero Cholesky decomposition elements
            x[np.abs(x) < 1e-10] += 1e-10

            # Reshape matrix
            indices = np.tril_indices(dimension)
            xL = np.zeros((dimension, dimension))
            xL[indices] = x

            # Compute SPD from Cholesky
            x = np.dot(xL, xL.T)

        base = vector_to_symmetric_matrix_mandel(base[0])

        # Projection in tangent space of the base
        x_proj = manifold.log(base, x)

        # Vectorize to use only once the symmetric elements
        # Division by sqrt(2) to keep the original elements (this is equivalent to Voigt instead of Mandel)
        x_proj_vec = symmetric_matrix_to_vector_mandel(x_proj)
        x_proj_vec[dimension:] /= 2. ** 0.5
        return x_proj_vec[None]

    elif isinstance(manifold, PositiveDefiniteProductEuclideanSphere):
        dimension = manifold.manifolds[0].dim
        x_eucl = x[0, 0:dimension]
        x_so = unit_sphere_to_rotation_matrix(x[0, dimension:])

        x_spd = np.dot(x_so, np.dot(np.diag(x_eucl), np.linalg.inv(x_so)))

        x_eucl_base = base[0, 0:dimension]
        x_so_base = unit_sphere_to_rotation_matrix(base[0, dimension:])
        base = np.dot(x_so_base, np.dot(np.diag(x_eucl_base), np.linalg.inv(x_so_base)))

        # Projection in tangent space of the base
        # We use the "classical" SPD manifold to do so, to have the same function.
        spd_manifold = SymmetricPositiveDefinite(dimension)
        x_proj = spd_manifold.log(base, x_spd)

        # Vectorize to use only once the symmetric elements
        # Division by sqrt(2) to keep the original elements (this is equivalent to Voigt instead of Mandel)
        x_proj_vec = symmetric_matrix_to_vector_mandel(x_proj)
        x_proj_vec[dimension:] /= 2. ** 0.5
        return x_proj_vec[None]

    elif isinstance(manifold, PositiveDefiniteProductEuclideanRotation):
        dimension = manifold.manifolds[0].dim
        x_eucl = x[0, 0:dimension]
        x_so = x[0, dimension:].reshape((dimension, dimension))

        x_spd = np.dot(x_so, np.dot(np.diag(x_eucl), np.linalg.inv(x_so)))

        x_eucl_base = base[0, 0:dimension]
        x_so_base = base[0, dimension:].reshape((dimension, dimension))
        base = np.dot(x_so_base, np.dot(np.diag(x_eucl_base), np.linalg.inv(x_so_base)))

        # Projection in tangent space of the base
        # We use the "classical" SPD manifold to do so, to have the same function.
        spd_manifold = SymmetricPositiveDefinite(dimension)
        x_proj = spd_manifold.log(base, x_spd)

        # Vectorize to use only once the symmetric elements
        # Division by sqrt(2) to keep the original elements (this is equivalent to Voigt instead of Mandel)
        x_proj_vec = symmetric_matrix_to_vector_mandel(x_proj)
        x_proj_vec[dimension:] /= 2. ** 0.5
        return x_proj_vec[None]

    elif isinstance(manifold, HyperbolicLorentz):
        x_proj = manifold.log(base, x)

        # With (1, 0, 0...) as base, the first component is always 0, so we remove it
        return x_proj[:, 1:]
        # return x_proj

    elif isinstance(manifold, Torus):
        x_list = [x[:, 2*i:2*i+2] for i in range(len(manifold.manifolds))]
        base_list = [base[:, 2*i:2*i+2] for i in range(len(manifold.manifolds))]
        x_proj = manifold.log(base_list, x_list)

        x_proj = np.array(list(x_proj))[:, 0, 1]
        # Remove first dim for all
        return x_proj[None]

    elif isinstance(manifold, SpecialOrthogonalGroup):
        # Dimension
        dimension = manifold._n

        base = base.reshape((dimension, dimension))
        x = x.reshape((dimension, dimension))
        x_proj = manifold.log(base, x)

        return x_proj.reshape(dimension**2)[None]

    else:
        raise RuntimeError("The preprocessing is not implemented for this manifold.")


class BenchmarkFunction:
    def __init__(self, manifold, cholesky=False):
        self.manifold = manifold

        if not isinstance(manifold, SymmetricPositiveDefinite):
            self.cholesky = False
        else:
            self.cholesky = cholesky

    def compute_function(self, x):
        pass

    def compute_function_torch(self, x):
        # Data to numpy
        torch_type = x.dtype
        x = x.cpu().detach().numpy()

        # Compute function
        y = self.compute_function(x)
        return torch.tensor(y, dtype=torch_type)

    def optimum(self):
        # Optimum x
        opt_x = get_base(self.manifold)
        # Optimum y
        opt_y = self.compute_function(opt_x)

        return opt_x, opt_y

    def get_base(self):
        return get_base(self.manifold)


class Ackley(BenchmarkFunction):
    def __init__(self, manifold):
        super(Ackley, self).__init__(manifold)

    def compute_function(self, x):
        if np.ndim(x) < 2:
            x = x[None]

        # Preprocess the input
        x = preprocess_manifold_data(self.manifold, x, self.cholesky)[0]

        # Dimension of the input
        dimension = x.shape[0]

        # Ackley function parameters
        a = 20
        b = 0.2
        c = 2 * np.pi

        # Ackley function
        aexp_term = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / dimension))
        expcos_term = - np.exp(np.sum(np.cos(c * x) / dimension))
        y = aexp_term + expcos_term + a + np.exp(1.)

        return y[None, None]


class Rosenbrock(BenchmarkFunction):
    def __init__(self, manifold):
        super(Rosenbrock, self).__init__(manifold)

    def compute_function(self, x):
        if np.ndim(x) < 2:
            x = x[None]

        # Preprocess the input
        x = preprocess_manifold_data(self.manifold, x, self.cholesky)[0]

        # Center optimum
        x = x + 1.

        # Rosenbrock function
        # y = 0
        # for i in range(reduced_dimension - 1):
        #     y += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        a = (x[1:] - x[:-1] ** 2)
        b = (1 - x[:-1])
        y = np.sum(100 * a * a + b * b)

        return y[None, None]


class Levy(BenchmarkFunction):
    def __init__(self, manifold, rescaling_factor=1.):
        super(Levy, self).__init__(manifold)

        if rescaling_factor is None:
            if isinstance(manifold, Sphere):
                self.rescaling_factor = 3.
            else:
                self.rescaling_factor = 1.
        else:
            self.rescaling_factor = rescaling_factor

    def compute_function(self, x):
        if np.ndim(x) < 2:
            x = x[None]

        # Preprocess the input
        x = preprocess_manifold_data(self.manifold, x, self.cholesky)[0]

        # Center optimum and rescale
        x = self.rescaling_factor * (x - 1.)

        # Dimension of the input
        dimension = x.shape[0]

        # Levy function
        pi = np.pi
        w1 = 1 + (x[0] - 1) / 4.
        y = np.sin(pi * w1) ** 2
        for i in range(dimension - 1):
            wi = 1 + (x[i] - 1) / 4.
            y += (wi - 1) ** 2 * (1 + 10 * np.sin(pi * wi + 1) ** 2)
        wd = 1 + (x[-1] - 1) / 4.
        y += (wd - 1) ** 2 * (1 + np.sin(2 * pi * wd) ** 2)

        return y[None, None]


class StyblinskiTang(BenchmarkFunction):
    def __init__(self, manifold, rescaling_factor=None):
        super(StyblinskiTang, self).__init__(manifold)
        if rescaling_factor is None:
            if isinstance(manifold, Sphere):
                self.rescaling_factor = 3.
            elif isinstance(manifold, SymmetricPositiveDefinite):
                self.rescaling_factor = 5.
            else:
                self.rescaling_factor = 1.
        else:
            self.rescaling_factor = rescaling_factor

    def compute_function(self, x):
        if np.ndim(x) < 2:
            x = x[None]

        # Preprocess the input
        x = preprocess_manifold_data(self.manifold, x, self.cholesky)[0]

        # Center the optimum and rescale
        x = self.rescaling_factor * x - 2.903534

        # Styblinski-tang function
        y = 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5. * x)

        return y[None, None]


class ProductOfSines(BenchmarkFunction):
    def __init__(self, manifold, coefficient=100.):
        super(ProductOfSines, self).__init__(manifold)
        self.coefficient = coefficient

    def compute_function(self, x):
        if np.ndim(x) < 2:
            x = x[None]

        # Preprocess the input
        x = preprocess_manifold_data(self.manifold, x, self.cholesky)[0]

        # Dimension of the input
        dimension = x.shape[0]

        # Center
        opt_x = np.pi / 2. * np.ones(dimension)
        opt_x[1] = - np.pi / 2.
        x = x + opt_x

        # Sinus
        sin_x = np.sin(x)
        # Product of sines function
        y = self.coefficient * sin_x[0] * np.prod(sin_x)

        return y[None, None]



