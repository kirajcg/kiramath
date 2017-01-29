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


def acf(L, h):
    return acvf(L, h) / acvf(L, 0)


def acvf(L, h):
    n = len(L)
    return sum([(L[t + abs(h)] - mean(L)) * (L[t] - mean(L)) for t in
                range(1, n - abs(h))])


# Moving-block bootstrap
def blockboot(L, b, B=1):
    N = len(L)
    Lsplit = [L[i:i+b] for i in range(N-b+1)]
    L_out = [0]*B
    for i in range(B):
        L_b = stats.sample(Lsplit, N//b)
        L_out[i] = [item for sublist in L_b for item in sublist]
    return L_out


def forecast(L, n=0):
    pass


def mean(L):
    n = len(L)
    return sum(L) / n


# Maximum entropy bootstrap
def meboot(L, J=1):
    N = len(L)
    # step 1: sort L and save indices
    L_sort = sorted((e,i) for i,e in enumerate(L))
    L_vals = [l[0] for l in L_sort]
    L_ind = [l[1] for l in L_sort]
    L_out = [0]*J
    # repetition
    for j in range(J):
        # step 2: compute intermediate points from order statistics
        Z = [(L_vals[i] + L_vals[i+1])/2 for i in range(N-1)]
        # step 3: extend intermediate points with endpoints
        m_trm = mean([abs(L[i] - L[i-1]) for i in range(1, N)])
        Z = [L_vals[0] - m_trm] + Z + [L_vals[-1] + m_trm]
        # step 4: compute mean of maximum entropy density
        m = [0]*N
        m[0] = 0.75*L_vals[0] + 0.25*L_vals[1]
        for k in range(1, N-1):
            m[k] = 0.25*L_vals[k-1] + 0.5*L_vals[k] + 0.25*L_vals[k+1]
        m[-1] = 0.25*L_vals[-2] + 0.75*L_vals[-1]
        # step 5: compute quantiles based on U(0, 1) nunmbers
        U = sorted(rand.random(n=N))
        quantiles = [0]*N
        # linear interpolation to find quantiles
        x = [float(y)/N for y in range(N+1)]
        for k in range(N):
            ind = min(range(len(x)), key=lambda i: abs(x[i] - U[k]))
            if x[ind] > U[k]:
                ind -= 1
            # mean-preservation
            c = (2*m[ind] - Z[ind] - Z[ind + 1]) / 2
            y0 = Z[ind] + c
            y1 = Z[ind + 1] + c
            quantiles[k] = y0 + (U[k] - x[ind]) * \
                            (y1 - y0) / (x[ind + 1] - x[ind])
        # step 6: reorder quantiles according to index vector
        L_out[j] = [x for y, x in sorted(zip(L_ind, quantiles))]
    return L_out


def pacf(L, h):
    if h == 1:
        return acf(L, 1)
    else:
        acfs = [acf(L, hh) for hh in range(h)]
        A = [acfs[i:] + acfs[:i] for i in range(h)]
        b = [acf(L, hh) for hh in range(h + 1)][1:]
        sol = linalg.solve(A, b)
        return sol[-1][0]
