# -*- coding: utf-8 -*-


def cos(x, N=2**7):
    y = 0
    for n in range(N + 1):
        y += (-1)**n / fac(2 * n) * x**(2 * n)
    return y


def exp(x, N=2**7):
    y = 0
    for n in range(N + 1):
        y += x**n / fac(n)
    return y


def log(x, N=2**7):
    y = 0
    if (abs(x - 1) <= 1):
        for n in range(1, N + 1):
            y += (-1)**(n + 1) / n * (x - 1)**n
        return y
    elif (x >= 1 / 2):
        for n in range(1, N + 1):
            y += 1 / n * ((x - 1) / x)**n
        return y
    else:
        raise ValueError("Math domain error")


def fac(n):
    return 1 if n == 0 else n * fac(n - 1)


def sin(x, N=2**7):
    y = 0
    for n in range(N + 1):
        y += (-1)**n / fac(2 * n + 1) * x**(2 * n + 1)
    return y


def sqrt(S, tol=1e-14):
    x = S
    while abs(x**2 - S) > tol:
        x = (x + S / x) / 2
    return x


def tan(x):
    return sin(x) / cos(x)
