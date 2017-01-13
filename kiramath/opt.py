# -*- coding: utf-8 -*-

from . import calc
from . import linalg


def gradient_descent(f, *args, **kwargs):
    if 'tol' not in kwargs:
        tol = 2e-16
    else:
        tol = kwargs['tol']

    if 'h' not in kwargs:
        h = 2**(-10)
    else:
        h = kwargs['h']

    x_old = list(args)
    x_new = [x + 1 for x in x_old]

    while linalg.norm([xo - xn for xo, xn in zip(x_old, x_new)]) > tol:
        x_old = x_new
        x_new = [x - h * g for x, g in zip(x_new, calc.gradient(f, *x_old))]

    return x_new, f(*x_new)


def newton_raphson(f, *args, **kwargs):
    if 'tol' not in kwargs:
        tol = 2e-16
    else:
        tol = kwargs['tol']

    if 'h' not in kwargs:
        h = 0.1
    else:
        h = kwargs['h']

    x = list(args)
    while linalg.norm(calc.gradient(f, *x)) > tol:
        l = linalg.matmult(linalg.inv(calc.hessian(f, *x)),
                           calc.gradient(f, *x))
        l_flat = [item for sublist in l for item in sublist]
        x = [xi - h * g for xi, g in zip(x, l_flat)]
    return x, f(*x)
