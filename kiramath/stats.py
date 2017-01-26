# -*- coding: utf-8 -*-

from . import elem
from . import rand


def bootstrap(X, B):
    n = len(X)
    boot = [sample(X, n, replacement=True) for _ in range(B)]
    return boot


def corr(L1, L2):
    return cov(L1, L2) / (std(L1) * std(L2))


def cov(L1, L2):
    n = len(L1)
    return (sum([l1 * l2 for l1, l2 in
            zip(L1, L2)]) - sum(L1) * sum(L2) / n) / n


def jackknife(X, J, n):
    jack = [sample(X, n, replacement=False) for _ in range(J)]
    return jack


def mad(L):
    m = median(L)
    shift = [abs(l - m) for l in L]
    return median(shift)


def mean(L):
    n = len(L)
    return sum(L) / n


def median(L):
    n = len(L)
    L_copy = L.copy()
    L_copy.sort()
    if n % 2 == 1:
        return L_copy[n//2]
    else:
        return mean([L_copy[n//2 - 1], L_copy[n//2]])


def sample(X, n, replacement=False):
    S = [0]*n
    X_copy = X.copy()
    for i in range(n):
        k = rand.randint(0, len(X_copy))[0]
        S[i] = X_copy[k]
        if replacement == False:
            X_copy.pop(k)
    return S


def shuffle(X):
    return sample(X, len(X), replacement=False)


def std(L):
    return elem.sqrt(var(L))


def var(L):
    n = len(L)
    return (sum([l**2 for l in L]) - sum(L)**2 / n) / (n - 1)
