import numpy as np
import torch
import itertools
from functools import reduce


def cartan_matrix_b(rank):
    c = np.zeros((rank, rank))
    for i in range(rank):
        for j in range(rank):
            if i == j:
                c[i, j] = 2.
            elif abs(i - j) == 1:
                c[i, j] = -1.

    c[rank - 2, rank - 1] = -2.
    return c


def cartan_matrix_d(rank):
    c = np.zeros((rank, rank))
    for i in range(rank):
        for j in range(rank):
            if i == j:
                c[i, j] = 2.
            elif abs(i - j) == 1:
                c[i, j] = -1.

    if rank > 2:
        c[rank - 3:rank, rank - 3:rank] = np.array([[2., -1., -1.], [-1., 2., 0.], [-1., 0., 2.]])
    return c


def cartran_matrix_so(n):
    if n == 2 or n == 3:
        return np.array([[2.]])
    else:
        k, r = divmod(n, 2)
        if r:
            return cartan_matrix_b(k)
        else:
            return cartan_matrix_d(k)


# quite general symmetrization function for Cartan matrices, to be replaced with a direct generation of Gram matrices
def symmetrize(m):
    rk = m.shape[0]
    gram = torch.zeros((rk, rk), dtype=m.dtype)
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
    # return reduce(lambda xx, yy: sp.lcm(xx, yy), map(lambda xx: xx.as_numer_denom()[1], set(flatten(gram)))) * gram
    # TODO: the line below works as long as the gram matrix elements are fractions of the form 1/d. Original implementation above.
    return reduce(lambda xx, yy: torch.lcm(xx, yy), torch.abs(1. / torch.flatten(gram)).int()) * gram


def b_func(x, y, bm):
    return torch.mm(x.transpose(-1, -2), torch.mm(bm, y))[0, 0]


def s_func(x, bm):
    return b_func(x, x, bm)


def ai_func(i, rank):
    a = torch.zeros((rank, 1)).double()
    a[i - 1] = 1
    return a


def wi_func(i, bm):
    rank = bm.shape[0]
    aii = ai_func(i, rank)
    saii = s_func(aii, bm)
    wm = torch.zeros((rank, rank), dtype=bm.dtype)
    for j in range(rank):
        wm[:, j] = (ai_func(j + 1, rank) - 2 * b_func(ai_func(j + 1, rank), aii, bm) / saii * aii)[:, 0]
    return wm