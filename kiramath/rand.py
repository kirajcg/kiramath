# -*- coding: utf-8 -*-

from . import datetime
from . import elem
from . import linalg
from . import spec


# Mersenne twister borrowed from Wikipedia
def _int32(x):
    # Get the 32 least significant bits.
    return int(0xFFFFFFFF & x)

class MT19937:

    def __init__(self, seed):
        # Initialize the index to 0
        self.index = 624
        self.mt = [0] * 624
        self.mt[0] = seed  # Initialize the initial state to the seed
        for i in range(1, 624):
            self.mt[i] = _int32(
                1812433253 * (self.mt[i - 1] ^ self.mt[i - 1] >> 30) + i)

    def extract_number(self):
        if self.index >= 624:
            self.twist()

        y = self.mt[self.index]

        # Right shift by 11 bits
        y = y ^ y >> 11
        # Shift y left by 7 and take the bitwise and of 2636928640
        y = y ^ y << 7 & 2636928640
        # Shift y left by 15 and take the bitwise and of y and 4022730752
        y = y ^ y << 15 & 4022730752
        # Right shift by 18 bits
        y = y ^ y >> 18

        self.index = self.index + 1

        return _int32(y)

    def twist(self):
        for i in range(624):
            # Get the most significant bit and add it to the less significant
            # bits of the next number
            y = _int32((self.mt[i] & 0x80000000) +
                       (self.mt[(i + 1) % 624] & 0x7fffffff))
            self.mt[i] = self.mt[(i + 397) % 624] ^ y >> 1

            if y % 2 != 0:
                self.mt[i] = self.mt[i] ^ 0x9908b0df
        self.index = 0


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
        MT = MT19937(seed=self.seed)
        U = [0]*n
        for i in range(n):
            self.set_seed(MT.extract_number())
            U[i] = self.seed / 0xFFFFFFFF
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

    def normal(self, mu=0, sigma=1, n=1, **kwargs):
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