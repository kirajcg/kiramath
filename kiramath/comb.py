# -*- coding: utf-8 -*-

from . import elem


def binom(n, k):
    if k > n:
        raise ValueError('%i greater than %i' % (k, n))
    return elem.fac(n) / (elem.fac(k) * elem.fac(n - k))
