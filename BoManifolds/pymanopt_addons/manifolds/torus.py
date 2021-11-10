from __future__ import division

from pymanopt.manifolds.sphere import Sphere
from pymanopt.manifolds.product import Product


class Torus(Product):

    def __init__(self, dimension):

        self.manifolds = [Sphere(2)] * dimension
        super(Torus, self).__init__(self.manifolds)
