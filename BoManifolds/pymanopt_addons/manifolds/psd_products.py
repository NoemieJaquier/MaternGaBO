"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier and Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
"""

from __future__ import division

from pymanopt.manifolds.euclidean import Euclidean
from pymanopt.manifolds.sphere import Sphere
from pymanopt.manifolds.special_orthogonal_group import SpecialOrthogonalGroup
from pymanopt.manifolds.product import Product


class PositiveDefiniteProductEuclideanSphere(Product):

    def __init__(self, dimension):

        if dimension == 2:
            dimension_sphere = 2
        elif dimension == 3:
            dimension_sphere = 4
        else:
            raise NotImplementedError

        self.manifolds = [Euclidean(dimension), Sphere(dimension_sphere)]
        super(PositiveDefiniteProductEuclideanSphere, self).__init__(self.manifolds)


class PositiveDefiniteProductEuclideanRotation(Product):

    def __init__(self, dimension):

        self.manifolds = [Euclidean(dimension), SpecialOrthogonalGroup(dimension)]
        super(PositiveDefiniteProductEuclideanRotation, self).__init__(self.manifolds)

