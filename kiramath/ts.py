# -*- coding: utf-8 -*-

from . import rand


def arma_model(L, p=0, q=0):
    pass


def arma_process(ar_coef=[1], ma_coef=[1], N=1, sigma=1):
    X = [0]*N
    Z = rand.normal(sigma=sigma, n=N)

    for i in range(N):
        X[i] += sum([ar_coef[j] * X[i - j] for j in range(len(ar_coef))])
        X[i] += sum([ma_coef[k] * Z[i - k] for k in range(len(ma_coef))])
    return X


def acf(L, h):
    return acvf(L, h) / acvf(L, 0)


def acvf(L, h):
    n = len(L)
    return sum([(L[t + abs(h)] - mean(L)) * (L[t] - mean(L)) for t in
                range(1, n - abs(h))])


def forecast(L, n=0):
    pass


def mean(L):
    n = len(L)
    return sum(L) / n


def pacf(L, h):
    pass
