# -*- coding: utf-8 -*-

from . import elem


def binom(n, k):
    if k > n:
        raise ValueError('%i greater than %i' % (k, n))
    return elem.fac(n) / (elem.fac(k) * elem.fac(n - k))


def multinom(*args):
    n_sum = 0
    n_prod = 1
    for i in range(len(args)):
        n_sum += args[i]
        n_prod *= elem.fac(args[i])
    return elem.fac(n_sum)/n_prod
