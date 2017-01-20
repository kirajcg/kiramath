# -*- coding: utf-8 -*-

from . import datetime
from . import elem
from . import linalg
from . import spec


class Random:

    def __init__(self, seed=None):
        if seed is None:
            self.seed = datetime.now().microsecond
        else:
            self.seed = seed

    def set_seed(self, seed):
        self.seed = seed

    def get_seed(self):
        return self.seed

    def random(self, n=1):
        a = 1664525
        c = 1013904223
        m = 2**32
        U = [0]*n
        for i in range(n):
            self.seed = (a * self.seed + c) % m
            U[i] = self.seed / m
        return U

    def bernoulli(self, p, n=1):
        U = self.random(n=n)
        B = [1 if u < p else 0 for u in U]
        return B

    def binomial(self, n, p, N=1):
        B = [sum(self.bernoulli(p, n)) for _ in range(N)]
        return B

    def exp(self, lparam=1, n=1):
        U = self.random(n=n)
        E = [-1 / lparam * elem.log(u) for u in U]
        return E

    def normal(self, mu=0, sigma=1, n=1):
        U = self.random(n=n)
        N = [mu + sigma * elem.sqrt(2) * spec.erfinv(2 * u - 1) for u in U]
        return N

    def normmult(self, mu=[0], Sigma=[[1]]):
        S = linalg.cholesky(Sigma)
        X = self.normal(n=len(mu))
        SX = linalg.matmult(S, X)
        SX_flat = [item for sublist in SX for item in sublist]
        return [sx + m for sx, m in zip(SX_flat, mu)]

    def poisson(self, lparam=1, n=1):
        pass

    # random integer in the range [a, b)
    def randint(self, a, b, n=1):
        U = self.random(n=n)
        R = [a + int((b - a)*u) for u in U]
        return R


_inst = Random()
get_seed = _inst.get_seed
set_seed = _inst.set_seed
random = _inst.random
bernoulli = _inst.bernoulli
exp = _inst.exp
normal = _inst.normal
normmult = _inst.normmult
poisson = _inst.poisson
randint = _inst.randint
