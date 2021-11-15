"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier and Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
"""

from __future__ import division

import warnings

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import Manifold


class HyperbolicLorentz(Manifold):
    """
    Manifold of shape (n1+1) x n2 x ... x nk tensors with Minkowski norm of -1,
    that is -x(0)^2 + x(1)^2 + x(2)^2 + ... + x(n)^2 = -1
    This manifold is an embedded submanifold of Euclidean space (the set of
    matrices of size (n1+1) x n2 x ... equipped with the usual trace inner product).
    The metric is such that the hyperbolic manifold is a semi-Riemannian
    submanifold of Minkowski space. This implementation is based on
    hyperbolicfactory.m from the Manopt MATLAB package.

    Examples:
    The hyperbolic space H^2, such that -x(0)^2 + x(1)^2+ x(2)^2 = -1:
    hyperbolic = HyperbolicLorentz(1)
    """

    def __init__(self, *shape):
        self._shape = shape
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        elif len(shape) == 1:
            name = "HyperbolicLorentz manifold of {}-vectors".format(*shape)
        elif len(shape) == 2:
            name = "HyperbolicLorentz manifold of {}x{} matrices".format(*shape)
        else:
            name = "HyperbolicLorentz manifold of shape " + str(shape) + " tensors"

        dimension = np.prod(shape)
        super().__init__(name, dimension)

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return np.prod(self._shape)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def inner_minkowski_columns(self, U, V):
        q = -U[0]*V[0] + np.sum(U[1:]*V[1:], axis=0)
        return q

    def inner(self, X, U, V):
        return float(np.sum(self.inner_minkowski_columns(U, V)))

    def norm(self, X, U):
        # Take the max against 0 to avoid imaginary results due to round-off resulting in small negative number
        return np.sqrt(np.maximum(self.inner(X, U, U), 0.))

    def dists(self, U, V):
        X = U - V
        mink_sqnorms = np.maximum(0., self.inner_minkowski_columns(X, X))
        mink_norms = np.sqrt(mink_sqnorms)
        d = 2 * np.arcsinh(.5 * mink_norms)
        return d

    def dist(self, U, V):
        return la.norm(self.dists(U, V))

    def proj(self, X, H):
        inners = self.inner_minkowski_columns(X, H)
        return H + X*inners

    def egrad2rgrad(self, X, egrad):
        egrad[0] = - egrad[0]
        return self.proj(X, egrad)

    def ehess2rhess(self, X, egrad, ehess, U):
        egrad[0] = - egrad[0]
        ehess[0] = - ehess[0]
        inners = self.inner_minkowski_columns(X, egrad)
        return self.proj(X, ehess + U*inners)

    def exp(self, X, U, t=1.):
        tU = t*U
        # Compute the individual Minkowski norms of the columns of U
        mink_inners = self.inner_minkowski_columns(tU, tU)
        mink_norms = np.sqrt(np.maximum(0, mink_inners))
        # Coefficients for the exponential.
        # For b, note that NaN's appear when an element of mink_norms is zero,
        # in which case the correct convention is to define sinh(0)/0 = 1.
        a = np.cosh(mink_norms)
        b = np.zeros(mink_norms.shape)
        b[mink_norms < 1e-16] = 1
        b[mink_norms >= 1e-16] = np.sinh(mink_norms)/mink_norms

        return X*a + tU*b

    def retr(self, X, U):
        return self.exp(X, U)

    def log(self, X, Y):
        d = self.dists(X, Y)
        a = d
        # a = d/np.sinh(d)
        a[d >= 1e-16] /= np.sinh(d[d >= 1e-16])
        a[d < 1e-16] = 1.
        return self.proj(X, Y*a)

    def rand(self):
        X1 = rnd.randn(*self._shape)
        x0 = np.sqrt(1. + np.sum(X1**2, 0))
        return np.concatenate((np.array([x0]), X1), axis=0)

    def randvec(self, X):
        return self._normalize(X, self.proj(X, rnd.randn(X.shape)))

    def transp(self, X, Y, U):
        return self.proj(Y, U)

    def pairmean(self, X, Y):
        return self.exp(X, self.log(X, Y), .5)

    def _normalize(self, X, U):
        return U / self.norm(X, U)

    def zerovec(self, X):
        return np.zeros(np.shape(X))

