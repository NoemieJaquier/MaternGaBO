"""
This file is part of the MaternGaBO library.
Authors: Andrei Smolensky, Viacheslav Borovitskiy, Noemie Jaquier, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu
"""

import numpy as np
import sympy as sp
import random
import itertools as it
import more_itertools as mit
import operator
from functools import reduce
import torch
import gpytorch
from gpytorch.constraints import Positive
import pymanopt.manifolds as manifolds

from BoManifolds.math_utils.so_root_systems import cartran_matrix_so, symmetrize, wi, s

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'
    

class SORiemannianGaussianKernel(gpytorch.kernels.Kernel):
    def __init__(self, dim, sigma_squared=1, summands_number_bound=10, lambda_coeff_bound=10 ** -10, **kwargs):

        self.has_lengthscale = True
        super(SORiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        self.dim = dim

        # Compute the symbolic expression of the kernel
        cm = cartran_matrix_so(dim)
        # the matrix of fundamental weights in the simple roots basis
        fwsm = cm.inv().transpose()
        rank = dim // 2
        # the Gram matrix of the bilinear form in the simple roots basis
        bm = symmetrize(cm)
        # delta = half-sum of positive roots = sum of fundamental weights
        delta = fwsm * sp.ones(rank, 1)
        # listing all elements of the Weyl group
        weyl_group = set()
        simple_reflections = [wi(i, bm) for i in range(1, rank + 1)]
        new_weyl = simple_reflections.copy()
        while new_weyl:
            weyl_group.update(new_weyl)
            prev_weyl = new_weyl
            new_weyl = set()
            for w in prev_weyl:
                for sr in simple_reflections:
                    s_weyl = sr * w
                    if s_weyl not in weyl_group:
                        new_weyl.add(s_weyl)

        # creating the list of all positive roots (can be used to calculate the dimension of the representation)
        # roots = {sp.ImmutableMatrix(rank, 1, lambda i, _: 1 if i == j else 0) for j in range(0, rank)}
        # roots.update(filter(lambda r: all(e >= 0 for e in r), [ss * r for r in roots for ss in weyl_group]))

        # helper function for the alternating sum
        def monomial(powers, vars):
            return reduce(operator.mul, (var ** deg for var, deg in zip(vars, powers)))

        # the alternating sum appearing in the numerator and the denominator of Weyl character formula
        def alt(weight, h):
            return sum(w.det() * monomial((w * weight).transpose() * bm, h) for w in weyl_group)

        kernel_summands = []
        total_norm_squared = 0
        h = sp.Matrix(sp.symbols(' '.join('x{}'.format(i) for i in range(1, rank + 1)))) if rank > 1 else [
            sp.symbols('x')]
        kappa = [sp.symbols('k')]
        for height in it.count(0):
            max_lambda_coeff = 0
            met_aiw = False
            for part in sp.utilities.iterables.partitions(height, m=rank):
                # translate partition dictionary into a tuple
                part = tuple(tuple(
                    mit.padded(it.chain.from_iterable((it.repeat(number, times) for number, times in part.items())), 0,
                               rank)))
                for pi in mit.distinct_permutations(part):
                    # pi is the heighest weight, expressed in the fundamental weights basis
                    # pi_vec is pi expressed in the roots basis
                    pi_vec = fwsm * sp.Matrix(rank, 1, pi)
                    # check if the weight pi in analytically integral
                    if all(map(operator.attrgetter('is_integer'), list(bm * pi_vec))):
                        met_aiw = True
                        # calculate Laplace-Beltrami eigenvalue corresponding
                        # to the irreducible representation with the highest weight pi
                        lb_ev = s(pi_vec + delta, bm) - s(delta, bm)
                        # chi = character of the representation, calculated via Weyl character formula
                        # as the polynomial in the eigenvalues of the input matrix.
                        # The simplification of the ratio of the factorized sums is done in order to avoid
                        # division by zero when evaluating at the identity matrix.
                        chi = sp.simplify(sp.factor(alt(pi_vec + delta, h)) / sp.factor(alt(delta, h)))
                        # the dimension of the representation calculated at the value of the character at 1
                        rep_dim = chi.subs([(var, 1) for var in h])
                        lambda_coeff = sp.exp(-lb_ev * kappa[0] ** 2 / 2) * rep_dim
                        # lambda_coeff = coeff_function(lb_ev) * rep_dim
                        total_norm_squared += lambda_coeff ** 2
                        # max_lambda_coeff = max(max_lambda_coeff, lambda_coeff)  # commented
                        # if lambda_coeff > lambda_coeff_bound: # commented
                        kernel_summands.append(lambda_coeff * chi)
                        print('hw = {}   dim = {}   coeff = {}'.format(pi, rep_dim, lambda_coeff.evalf()))
            # if max_lambda_coeff < lambda_coeff_bound and met_aiw or len(kernel_summands) >= summands_number_bound:  # commented
            if len(kernel_summands) >= summands_number_bound:
                break
        kernel = sigma_squared / total_norm_squared * sum(kernel_summands)
        print('kernel = {}'.format(kernel))
        print('num. reps = {}'.format(len(kernel_summands)))
        print('total sq. norm = {}'.format(total_norm_squared.evalf()))

        # Save the serie
        array2mat = [{'ImmutableDenseMatrix': torch.tensor}, 'torch']
        self.series_approx = sp.lambdify(list(h)+[kappa], kernel, modules=array2mat)

    def forward(self, x1, x2, diag=False, **params):
        # Reshape matrices
        x1_shape = list(x1.shape)[:-1] + [self.dim, self.dim]
        x2_shape = list(x2.shape)[:-1] + [self.dim, self.dim]
        x1 = x1.view(x1_shape)
        x2 = x2.view(x2_shape)

        # Expand dimensions to compute all matrix-matrix relationships
        x1 = x1.unsqueeze(-3)
        x2 = x2.unsqueeze(-4)
        # Repeat x and y data along 1- and 0- dimensions to have ndata_x x ndata_y x dim x dim arrays
        x1 = torch.cat(x2.shape[-3] * [x1], dim=-3)
        x2 = torch.cat(x1.shape[-4] * [x2], dim=-4)

        kernel_shape = list(x1.shape)[:-2]

        error = 10 ** -10

        if self.dim == 3:
            matrix_ratio = torch.bmm(x1.view(-1, self.dim, self.dim), x2.view(-1, self.dim, self.dim).transpose(-1, -2))
            imaginary_part = torch.arccos(torch.clip((matrix_ratio.diagonal(dim1=-1, dim2=-2).sum(-1)-1.)/2., -1.+1e-8, 1.-1e-8))
            exph = torch.exp(torch.complex(torch.zeros_like(imaginary_part), imaginary_part))
            kernel = self.series_approx(exph, self.lengthscale).real

        else:
            eigenv = torch.linalg.eigvals(torch.bmm(x1.view(-1, self.dim, self.dim),
                                                    x2.view(-1, self.dim, self.dim).transpose(-1, -2)))

            # Compute kernel
            kernel = torch.zeros(eigenv.shape[0], dtype=x1.dtype).to(device)
            for k in range(eigenv.shape[0]):
                eigenv_list = list(eigenv[k])
                exph = []
                while eigenv_list:
                    ev = eigenv_list.pop(0)
                    evc = torch.conj(ev)
                    for i, a in enumerate(eigenv_list):
                        if torch.abs(evc - a) < error:
                            exph.append(ev[None])
                            del eigenv_list[i]
                            break
                kernel[k] = self.series_approx(*exph, self.lengthscale).real

        # Compute normalizing term
        eigenv_norm = torch.ones(self.dim, dtype=x1.dtype).to(device)
        eigenv_norm_list = list(eigenv_norm)
        exph = []
        while eigenv_norm_list:
            ev = eigenv_norm_list.pop(0)
            evc = torch.conj(ev)
            for i, a in enumerate(eigenv_norm_list):
                if torch.abs(evc - a) < error:
                    exph.append(ev[None])
                    del eigenv_norm_list[i]
                    break
        norm_factor = self.series_approx(*exph, self.lengthscale)  #.real

        return kernel.view(kernel_shape) / norm_factor


class SORiemannianMaternKernel(gpytorch.kernels.Kernel):
    def __init__(self, dim, nu=None, nu_prior=None, sigma_squared=1, summands_number_bound=10,
                 lambda_coeff_bound=10 ** -10, **kwargs):

        self.has_lengthscale = True
        super(SORiemannianMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

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

        self.dim = dim

        # Compute the symbolic expression of the kernel
        cm = cartran_matrix_so(dim)
        # the matrix of fundamental weights in the simple roots basis
        fwsm = cm.inv().transpose()
        rank = dim // 2
        # the dimension of SO(n) depending on the rank and the classical series (B or D)
        so_dim = 2 * rank ** 2 + rank if dim % 2 else 2 * rank ** 2 - rank
        # the Gram matrix of the bilinear form in the simple roots basis
        bm = symmetrize(cm)
        # delta = half-sum of positive roots = sum of fundamental weights
        delta = fwsm * sp.ones(rank, 1)
        # listing all elements of the Weyl group
        weyl_group = set()
        simple_reflections = [wi(i, bm) for i in range(1, rank + 1)]
        new_weyl = simple_reflections.copy()
        while new_weyl:
            weyl_group.update(new_weyl)
            prev_weyl = new_weyl
            new_weyl = set()
            for w in prev_weyl:
                for sr in simple_reflections:
                    s_weyl = sr * w
                    if s_weyl not in weyl_group:
                        new_weyl.add(s_weyl)

        # creating the list of all positive roots (can be used to calculate the dimension of the representation)
        # roots = {sp.ImmutableMatrix(rank, 1, lambda i, _: 1 if i == j else 0) for j in range(0, rank)}
        # roots.update(filter(lambda r: all(e >= 0 for e in r), [ss * r for r in roots for ss in weyl_group]))

        # helper function for the alternating sum
        def monomial(powers, vars):
            return reduce(operator.mul, (var ** deg for var, deg in zip(vars, powers)))

        # the alternating sum appearing in the numerator and the denominator of Weyl character formula
        def alt(weight, h):
            return sum(w.det() * monomial((w * weight).transpose() * bm, h) for w in weyl_group)

        kernel_summands = []
        total_norm_squared = 0
        h = sp.Matrix(sp.symbols(' '.join('x{}'.format(i) for i in range(1, rank + 1)))) if rank > 1 else [
            sp.symbols('x')]
        kappa = [sp.symbols('k')]
        smoothness = [sp.symbols('nu')]
        for height in it.count(0):
            max_lambda_coeff = 0
            met_aiw = False
            for part in sp.utilities.iterables.partitions(height, m=rank):
                # translate partition dictionary into a tuple
                part = tuple(tuple(
                    mit.padded(it.chain.from_iterable((it.repeat(number, times) for number, times in part.items())), 0,
                               rank)))
                for pi in mit.distinct_permutations(part):
                    # pi is the heighest weight, expressed in the fundamental weights basis
                    # pi_vec is pi expressed in the roots basis
                    pi_vec = fwsm * sp.Matrix(rank, 1, pi)
                    # check if the weight pi in analytically integral
                    if all(map(operator.attrgetter('is_integer'), list(bm * pi_vec))):
                        met_aiw = True
                        # calculate Laplace-Beltrami eigenvalue corresponding
                        # to the irreducible representation with the highest weight pi
                        lb_ev = s(pi_vec + delta, bm) - s(delta, bm)
                        # chi = character of the representation, calculated via Weyl character formula
                        # as the polynomial in the eigenvalues of the input matrix.
                        # The simplification of the ratio of the factorized sums is done in order to avoid
                        # division by zero when evaluating at the identity matrix.
                        chi = sp.simplify(sp.factor(alt(pi_vec + delta, h)) / sp.factor(alt(delta, h)))
                        # the dimension of the representation calculated at the value of the character at 1
                        rep_dim = chi.subs([(var, 1) for var in h])
                        lambda_coeff = sp.exp(-lb_ev * kappa[0] ** 2 / 2) * rep_dim
                        lambda_coeff = (2 * smoothness[0] / kappa[0] ** 2 + lb_ev) ** (-smoothness[0] - so_dim / 2) * rep_dim
                        # lambda_coeff = coeff_function(lb_ev) * rep_dim
                        total_norm_squared += lambda_coeff ** 2
                        # max_lambda_coeff = max(max_lambda_coeff, lambda_coeff)  # commented
                        # if lambda_coeff > lambda_coeff_bound: # commented
                        kernel_summands.append(lambda_coeff * chi)
                        print('hw = {}   dim = {}   coeff = {}'.format(pi, rep_dim, lambda_coeff.evalf()))
            # if max_lambda_coeff < lambda_coeff_bound and met_aiw or len(kernel_summands) >= summands_number_bound:  # commented
            if len(kernel_summands) >= summands_number_bound:
                break
        kernel = sigma_squared / total_norm_squared * sum(kernel_summands)
        print('kernel = {}'.format(kernel))
        print('num. reps = {}'.format(len(kernel_summands)))
        print('total sq. norm = {}'.format(total_norm_squared.evalf()))

        # Save the serie
        array2mat = [{'ImmutableDenseMatrix': torch.tensor}, 'torch']
        self.series_approx = sp.lambdify(list(h)+[kappa, smoothness], kernel, modules=array2mat)

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
        # Reshape matrices
        x1_shape = list(x1.shape)[:-1] + [self.dim, self.dim]
        x2_shape = list(x2.shape)[:-1] + [self.dim, self.dim]
        x1 = x1.view(x1_shape)
        x2 = x2.view(x2_shape)

        # Expand dimensions to compute all matrix-matrix relationships
        x1 = x1.unsqueeze(-3)
        x2 = x2.unsqueeze(-4)
        # Repeat x and y data along 1- and 0- dimensions to have ndata_x x ndata_y x dim x dim arrays
        x1 = torch.cat(x2.shape[-3] * [x1], dim=-3)
        x2 = torch.cat(x1.shape[-4] * [x2], dim=-4)

        kernel_shape = list(x1.shape)[:-2]

        error = 10 ** -10

        if self.dim == 3:
            matrix_ratio = torch.bmm(x1.view(-1, self.dim, self.dim), x2.view(-1, self.dim, self.dim).transpose(-1, -2))
            imaginary_part = torch.arccos(torch.clip((matrix_ratio.diagonal(dim1=-1, dim2=-2).sum(-1)-1.)/2., -1.+1e-8, 1.-1e-8))
            exph = torch.exp(torch.complex(torch.zeros_like(imaginary_part), imaginary_part))
            kernel = self.series_approx(exph, self.lengthscale, self.nu).real

        else:
            eigenv = torch.linalg.eigvals(torch.bmm(x1.view(-1, self.dim, self.dim),
                                                    x2.view(-1, self.dim, self.dim).transpose(-1, -2)))

            # Compute kernel
            kernel = torch.zeros(eigenv.shape[0], dtype=x1.dtype).to(device)
            for k in range(eigenv.shape[0]):
                eigenv_list = list(eigenv[k])
                exph = []
                while eigenv_list:
                    ev = eigenv_list.pop(0)
                    evc = torch.conj(ev)
                    for i, a in enumerate(eigenv_list):
                        if torch.abs(evc - a) < error:
                            exph.append(ev[None])
                            del eigenv_list[i]
                            break
                kernel[k] = self.series_approx(*exph, self.lengthscale, self.nu).real

        # Compute normalizing term
        eigenv_norm = torch.ones(self.dim, dtype=x1.dtype).to(device)
        eigenv_norm_list = list(eigenv_norm)
        exph = []
        while eigenv_norm_list:
            ev = eigenv_norm_list.pop(0)
            evc = torch.conj(ev)
            for i, a in enumerate(eigenv_norm_list):
                if torch.abs(evc - a) < error:
                    exph.append(ev[None])
                    del eigenv_norm_list[i]
                    break
        norm_factor = self.series_approx(*exph, self.lengthscale, self.nu) #.real

        return kernel.view(kernel_shape) / norm_factor


if __name__ == "__main__":
    # testing positive definiteness
    random.seed(1234)
    np.random.seed(1234)

    for dim in range(2, 7):
        rank = dim // 2
        so_dim = 2 * rank ** 2 + rank if dim % 2 else 2 * rank ** 2 - rank
        print('\ntesting dim={} (dimSO={}, rank={})'.format(dim, so_dim, rank))
        manifold = manifolds.SpecialOrthogonalGroup(dim)
        # kernel = SORiemannianGaussianKernel(dim, 1, summands_number_bound=10)
        kernel = SORiemannianMaternKernel(dim, nu=2.5, summands_number_bound=10)

        q = torch.from_numpy(np.array([manifold.rand() for i in range(10)])).view(-1, dim**2)
        q0 = q[0][None]
        K = kernel.forward(q0, q)

        # kernel = SOMaternKernel(dim, 1, 1, summands_number_bound=20)
        sizes_results = {}
        for size in range(2, 21):
            orth_matrices = []
            for item in range(size):
                q = torch.from_numpy(manifold.rand()).unsqueeze(0).view(-1, dim**2)
                orth_matrices.append(q)
            gram_matrix = [[kernel.forward(orth_matrices[i], orth_matrices[j]) for j in range(size)] for i in
                                  range(size)]  # + 10**-20 * np.eye(size)
            gram_matrix = torch.tensor(gram_matrix).numpy()


            try:
                np.linalg.cholesky(gram_matrix)
                sizes_results[size] = 1
            except np.linalg.LinAlgError as inst:
                sizes_results[size] = 0
            print('for {} elements: {} (min eigenvalue {})'.format(size, 'ok' if sizes_results[size] else 'NO',
                                                                   min(np.linalg.eigvalsh(gram_matrix))))
