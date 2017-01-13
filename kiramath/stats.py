# -*- coding: utf-8 -*-

from . import elem
from . import rand


def bootstrap(X, B):
    n = len(X)
    boot = [0]*B
    for i in range(B):
        boot[i] = sample(X, n, replacement=True)
    return boot


def corr(L1, L2):
    return cov(L1, L2) / (std(L1) * std(L2))


def cov(L1, L2):
    n = len(L1)
    return (sum([l1 * l2 for l1, l2 in
            zip(L1, L2)]) - sum(L1) * sum(L2) / n) / n


def jackknife(X, J, n):
    jack = [0]*J
    for i in range(J):
        jack[i] = sample(X, n, replacement=False)
    return jack


def mean(L):
    n = len(L)
    return sum(L) / n


def sample(X, n, replacement=False):
    S = [0]*n
    X_copy = X.copy()
    for i in range(n):
        k = rand.randint(0, len(X_copy))[0]
        S[i] = X_copy[k]
        if replacement == False:
            X_copy.pop(k)
    return S


def std(L):
    return elem.sqrt(var(L))


def var(L):
    n = len(L)
    return (sum([l**2 for l in L]) - sum(L)**2 / n) / (n - 1)
