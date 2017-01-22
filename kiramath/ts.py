# -*- coding: utf-8 -*-

from . import rand
from . import linalg
from . import stats


def arma_model(L, p=0, q=0):
    pass


def arma_process(ar_coef=[1], ma_coef=[1], N=1, sigma=1):
    X = [0]*N
    Z = rand.normal(sigma=sigma, n=N)

    for i in range(N):
        X[i] += sum([ar_coef[j] * X[i - j] for j in range(len(ar_coef))])
        X[i] += sum([ma_coef[k] * Z[i - k] for k in range(len(ma_coef))])
    return X


def acf(L, h_max):
    # returns acf for 0, ..., h_max - 1
    return [acvf(L, h) / acvf(L, 0) for h in range(h_max)]


def acvf(L, h):
    n = len(L)
    return sum([(L[t + abs(h)] - mean(L)) * (L[t] - mean(L)) for t in
                range(1, n - abs(h))])


# Moving-block bootstrap
def blockboot(L, b):
    N = len(L)
    Lsplit = [L[i:i+b] for i in range(N-b+1)]
    L_out = stats.sample(Lsplit, N//b)
    return [item for sublist in L_out for item in sublist]


def forecast(L, n=0):
    pass


def mean(L):
    n = len(L)
    return sum(L) / n


def pacf(L, h_max):
    acfs = acf(L, h_max)
    A = [acfs[i:] + acfs[:i] for i in range(h_max)]
    b = acf(L, h_max + 1)[1:]
    sol = linalg.solve(A, b)
    # return pacf for 0, ..., h_max - 1 as is pythonesque
    return acf(L, 1) + [item for sublist in sol for item in sublist][:-1]
