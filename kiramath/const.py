# -*- coding: utf-8 -*-


def pi(N=2**7):
    pi = 0
    for n in range(0, N + 1):
        pi += 1 / (16**n) * (4 / (8 * n + 1) - 2 / (8 * n + 4) -
                             1 / (8 * n + 5) - 1 / (8 * n + 6))
    return pi
