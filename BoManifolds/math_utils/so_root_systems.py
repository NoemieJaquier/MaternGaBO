"""
This file is part of the MaternGaBO library.
Authors: Andrei Smolensky, Viacheslav Borovitskiy, Noemie Jaquier, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu
"""

import sympy as sp
from sympy.utilities.iterables import flatten
import itertools
from functools import reduce


def cartan_matrix_b(rank):
    c = sp.Matrix(rank, rank, lambda i, j: 2 if i == j else -1 if abs(i - j) == 1 else 0)
    c[rank - 2, rank - 1] = -2
    return c


def cartan_matrix_d(rank):
    c = sp.Matrix(rank, rank, lambda i, j: 2 if i == j else -1 if abs(i - j) == 1 else 0)
    if rank > 2:
        c[rank - 3:rank, rank - 3:rank] = sp.Matrix([[2, -1, -1], [-1, 2, 0], [-1, 0, 2]])
    return c


def cartran_matrix_so(n):
    if n == 2 or n == 3:
        return sp.Matrix([[2]])
    else:
        k, r = divmod(n, 2)
        if r:
            return cartan_matrix_b(k)
        else:
            return cartan_matrix_d(k)


# quite general symmetrization function for Cartan matrices, to be replaced with a direct generation of Gram matrices
def symmetrize(m):
    rk = m.shape[0]
    gram = sp.zeros(rk, rk)
    unnormed_roots = set(range(rk))
    normed_roots = set()
    while unnormed_roots:
        x = unnormed_roots.pop()
        gram[x, x] = 1
        normed_roots.add(x)
        while unnormed_roots:
            new_normed = set()
            for x in normed_roots:
                for y in filter(lambda z: m[x, z] != 0, range(rk)):
                    gram[y, y] = gram[x, x] * m[y, x] / m[x, y]
                    new_normed.add(y)
            normed_roots.update(new_normed)
            unnormed_roots.difference_update(new_normed)
            if not new_normed:
                break
    for x, y in itertools.product(range(rk), repeat=2):
        gram[x, y] = m[x, y] * gram[y, y] / 2
    return reduce(lambda xx, yy: sp.lcm(xx, yy), map(lambda xx: xx.as_numer_denom()[1], set(flatten(gram)))) * gram


def b(x, y, bm):
    return (x.transpose() * bm * y)[0, 0]


def s(x, bm):
    return b(x, x, bm)


def ai(i, rank):
    a = sp.zeros(rank, 1)
    a[i - 1] = 1
    return a


def wi(i, bm):
    rank = bm.shape[0]
    aii = ai(i, rank)
    saii = s(aii, bm)
    wm = sp.zeros(rank, rank)
    for j in range(rank):
        wm[:, j] = ai(j + 1, rank) - 2 * b(ai(j + 1, rank), aii, bm) / saii * aii
    return sp.ImmutableMatrix(wm)