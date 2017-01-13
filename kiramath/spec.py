# -*- coding: utf-8 -*-

from . import calc
from . import const
from . import elem


def bernoulli(n):
    A = [0] * (n+1)
    for m in range(n+1):
        A[m] = 1/(m+1)
        for j in range(m, 0, -1):
          A[j-1] = j*(A[j-1] - A[j])
    return A[0]


def erf(x):
    expquad = lambda t: elem.exp(-t**2)
    pi = const.pi()
    erfx = 2 / elem.sqrt(pi) * calc.integrate(expquad, 0, x)
    return erfx


def erfinv(z, N=2**4):
    pi = const.pi()
    c = [0] * N
    erfinvz = 0
    c[0] = 1
    for k in range(1, N):
        for m in range(k):
            c[k] += c[m] * c[k - 1 - m] / ((m + 1) * (2 * m + 1))
    for k in range(N):
        erfinvz += c[k] / (2 * k + 1) * (elem.sqrt(pi) / 2 * z) ** (2 * k + 1)
    return erfinvz


def gamma(z, N=2**2):
    pi = const.pi()
    lngamma = z * elem.log(z) - z + 1 / 2 * elem.log(2 * pi / z) + \
                sum([bernoulli(2*n) / (2 * n * (2 * n - 1) * z**(2 * n - 1))
                for n in range(1, N)])
    return elem.exp(lngamma)
