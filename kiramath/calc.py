# -*- coding: utf-8 -*-

from . import linalg


def diff(f, x0, h=2**(-7)):
    deriv = (-f(x0 + 2 * h) + 8 * f(x0 + h) - 8 *
             f(x0 - h) + f(x0 - 2 * h)) / (12 * h)
    return deriv


def gradient(f, *args, **kwargs):
    if 'h' not in kwargs:
        h = 2**(-7)
    else:
        h = kwargs['h']

    grad = [0] * len(args)
    x0 = list(args)

    for i in range(len(x0)):
        x0[i] += h
        fplush = f(*x0)
        x0[i] -= 2 * h
        fminush = f(*x0)
        x0[i] += h
        grad[i] = (fplush - fminush) / (2 * h)
    return grad


def hessian(f, *args, **kwargs):
    if 'h' not in kwargs:
        h = 2**(-7)
    else:
        h = kwargs['h']

    H = [0] * len(args)
    x0 = list(args)

    for i in range(len(x0)):
        x0[i] += h
        dfplush = gradient(f, *x0)
        x0[i] -= 2 * h
        dfminush = gradient(f, *x0)
        x0[i] += h
        H[i] = [(d2 - d1) / (2 * h) for d2, d1 in zip(dfplush, dfminush)]
    return H


def integrate(f, a, b, n=2**7):
    integral = f(a) / 2 + f(b) / 2
    for k in range(1, n):
        integral += f(a + k * (b - a) / n)
    integral = integral * (b - a) / n
    return integral


def jacobian(f, *args):
    J = [0] * len(f)
    x0 = list(args)

    for i in range(len(f)):
        J[i] = gradient(f[i], *x0)

    return J


def ode(f, t0, y0, t1, n):
    t = [t0] * (n + 1)
    y = [y0] * (n + 1)
    h = (t1 - t0) / n
    for i in range(1, n + 1):
        k1 = f(t[i - 1], y[i - 1])
        k2 = f(t[i - 1] + h / 2, y[i - 1] + h / 2 * k1)
        k3 = f(t[i - 1] + h / 2, y[i - 1] + h / 2 * k2)
        k4 = f(t[i - 1] + h, y[i - 1] + h * k3)
        y[i] = y[i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t[i] = t[i - 1] + h
    return t, y


def zeros(f, x, h=0.1, tol=2e-16):
    while abs(f(x)) > tol:
        x = x - f(x) / diff(f, x)
    return x


def zeros_mult(f, *args, **kwargs):
    if 'tol' not in kwargs:
        tol = 2e-16
    else:
        tol = kwargs['tol']

    if 'h' not in kwargs:
        h = 0.1
    else:
        h = kwargs['h']

    x = list(args)
    fx = [func(*x) for func in f]
    while linalg.norm(fx) > tol:
        l = linalg.matmult(linalg.inv(jacobian(f, *x)), fx)
        l_flat = [item for sublist in l for item in sublist]
        x = [xi - h * g for xi, g in zip(x, l_flat)]
        fx = [func(*x) for func in f]
    return x
