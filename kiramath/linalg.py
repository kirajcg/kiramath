# -*- coding: utf-8 -*-

from . import elem


def adj(A):
    if len(A) == 2:
        return [[A[1][1], -1 * A[0][1]], [-1 * A[1][0], A[0][0]]]

    cf = [[(-1)**(r + c) * det(minor(A, r, c)) for c in range(len(A))]
          for r in range(len(A))]
    return transpose(cf)


def cholesky(A):
    L = [[0.0] * len(A) for _ in range(len(A))]
    for i, (Ai, Li) in enumerate(zip(A, L)):
        for j, Lj in enumerate(L[:i + 1]):
            s = sum(Li[k] * Lj[k] for k in range(j))
            Li[j] = elem.sqrt(Ai[i] - s) if (i == j) else \
                (1.0 / Lj[j] * (Ai[j] - s))
    return L


def det(A):
    if len(A) == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    else:
        return sum([(-1)**c * A[0][c] * det(minor(A, 0, c))
                    for c in range(len(A))])


def dot(x, y):
    return sum([x_i * y_i for x_i, y_i in zip(x, y)])


def eig(A, tol=1e-6):
    """ Returns highest eigenvalue and corresponding eigenvector """
    b = [1 / len(A) for i in range(len(A))]
    b = [sum([A[i][j] * b[j] for j in range(len(A))]) for i in range(len(A))]
    normb = norm(b)
    b = [b[i] / normb for i in range(len(b))]
    r = sum([sum([b[j] * A[i][j] * b[j] for j in range(len(A))])
             for i in range(len(A))])
    r_old = 0

    while abs(r - r_old) > tol:
        b = [sum([A[i][j] * b[j] for j in range(len(A))])
             for i in range(len(A))]
        normb = norm(b)
        b = [b[i] / normb for i in range(len(b))]
        r_old = r
        r = sum([sum([b[j] * A[i][j] * b[j] for j in range(len(A))])
                 for i in range(len(A))])

    return normb, b


def inv(A):
    return [[a / det(A) for a in adj(A)[i]] for i in range(len(adj(A)))]


def matmult(A, B):
    try:
        zip_B = list(zip(*B))
    except TypeError:
        zip_B = [list(B)]
    return [[sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b))
             for col_b in zip_B] for row_a in A]


def minor(A, i, j):
    return [row[:j] + row[j + 1:] for row in (A[:i] + A[i + 1:])]


def norm(A, p=1, tol=1e-6):
    if not isinstance(A[0], list):
        return elem.sqrt(sum([A[i] * A[i] for i in range(len(A))]), tol=tol)
    else:
        if p == 1:
            return max([sum([abs(A[i][j]) for i in range(len(A))])
                        for j in range(len(A[0]))])
        elif p == 'inf':
            return max([sum([abs(A[i][j]) for j in range(len(A[0]))])
                        for i in range(len(A))])
        elif p == 'frobenius':
            return sum([sum([abs(A[i][j])**2 for i in range(len(A))])
                        for j in range(len(A[0]))])


def solve(A, b):
    return matmult(inv(A), b)


def transpose(A):
    return [list(zip_a) for zip_a in list(zip(*A))]
