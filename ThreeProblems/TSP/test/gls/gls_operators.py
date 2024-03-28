import itertools

import numpy as np
from numba import jit

@jit(nopython=True)
def two_opt(tour, i, j):
    if i == j:
        return tour
    a = tour[i,0]
    b = tour[j,0]
    tour[i,0] = tour[i,1]
    tour[i,1] = j
    tour[j,0] = i
    tour[a,1] = b
    tour[b,1] = tour[b,0] 
    tour[b,0] = a
    c = tour[b,1]
    while tour[c,1] != j:
        d = tour[c,0]
        tour[c,0] = tour[c,1]
        tour[c,1] = d
        c = d
    return tour

@jit(nopython=True)
def two_opt_cost(tour, D, i, j):
    if i == j:
        return 0

    a = tour[i,0]
    b = tour[j,0]
    #print(D[a, c])
    delta = D[a, b] \
            + D[i, j] \
            - D[a, i] \
            - D[b, j]
    return delta

@jit(nopython=True)
def two_opt_a2a(tour, D,N, first_improvement=False, set_delta=0):
    best_move = None
    best_delta = set_delta

    idxs = range(0, len(tour) - 1)
    for i in idxs:
        for j in N[i]:
            if i in tour[j] or j in tour[i]:
                continue

            delta = two_opt_cost(tour, D, i, j)
            if delta < best_delta and not np.isclose(0, delta):
                best_delta = delta
                best_move = i, j
                if first_improvement:
                    break

    if best_move is not None:
        return best_delta, two_opt(tour, *best_move)
    return 0, tour

@jit(nopython=True)
def two_opt_o2a(tour, D, i, first_improvement=False):
    assert i > 0 and i < len(tour) - 1

    best_move = None
    best_delta = 0

    idxs = range(1, len(tour) - 1)
    for j in idxs:
        if abs(i - j) < 2:
            continue

        delta = two_opt_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break

    if best_move is not None:
        return best_delta, two_opt(tour, *best_move)
    return 0, tour

@jit(nopython=True)
def two_opt_o2a_all(tour, D,N, i):

    best_move = None
    best_delta = 0

    idxs = N[i]
    for j in idxs:
        if i in tour[j] or j in tour[i]:
            continue
        delta = two_opt_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            tour = two_opt(tour, *best_move)

    return best_delta , tour


@jit(nopython=True)
def relocate(tour, i, j):
    a = tour[i,0]
    b = tour[i,1]
    tour[a,1] = b
    tour[b,0] = a

    d = tour[j,1]
    tour[d,0] = i
    tour[i,0] = j
    tour[i,1] = d
    tour[j,1] = i

    return tour

@jit(nopython=True)
def relocate_cost(tour, D, i, j):
    if i == j:
        return 0

    a = tour[i,0]
    b = i
    c = tour[i,1]

    d = j
    e = tour[j,1]

    delta = - D[a, b] \
            - D[b, c] \
            + D[a, c] \
            - D[d, e] \
            + D[d, b] \
            + D[b, e]
    return delta

@jit(nopython=True)
def relocate_o2a(tour, D, i, first_improvement=False):
    assert i > 0 and i < len(tour) - 1

    best_move = None
    best_delta = 0

    idxs = range(1, len(tour) - 1)
    for j in idxs:
        if i == j:
            continue

        delta = relocate_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break

    if best_move is not None:
        return best_delta, relocate(tour, *best_move)
    return 0, tour

@jit(nopython=True)
def relocate_o2a_all(tour, D,N, i):
    best_move = None
    best_delta = 0

    for j in N[i]:
        if tour[j,1] == i:  # e.g. relocate 2 -> 3 == relocate 3 -> 2
            continue

        delta = relocate_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            tour = relocate(tour, *best_move)

    return best_delta, tour

@jit(nopython=True)
def relocate_a2a(tour, D,N, first_improvement=False, set_delta=0):
    best_move = None
    best_delta = set_delta

    idxs = range(0, len(tour) - 1)
    for i in idxs:
        for j in N[i]:
            if tour[j,1] == i:  # e.g. relocate 2 -> 3 == relocate 3 -> 2
                continue

            delta = relocate_cost(tour, D, i, j)
            if delta < best_delta and not np.isclose(0, delta):
                best_delta = delta
                best_move = i, j
                if first_improvement:
                    break

    if best_move is not None:
        return best_delta, relocate(tour, *best_move)
    return 0, tour
