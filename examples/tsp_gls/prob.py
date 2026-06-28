# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import time
import sys
import os
import random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


# ── GLS engine (pure-Python port of the original numba implementation) ──────────
#
# The tour is stored in a "2-end" representation: route2End[i] = (prev, next),
# i.e. for node i, route2End[i, 0] is its predecessor and route2End[i, 1] its
# successor in the cycle. Local search (2-opt + relocate) and the guided
# perturbation operate directly on this representation, restricted to each
# node's k-nearest-neighbour candidate list.


def tour_cost_2End(dis_m, tour2End):
    c = 0
    s = 0
    e = tour2End[0, 1]
    for _ in range(tour2End.shape[0]):
        c += dis_m[s, e]
        s = e
        e = tour2End[s, 1]
    return c


def nearest_neighbor_2End(dis_matrix, depot):
    tour = [depot]
    n = len(dis_matrix)
    nodes = np.arange(n)
    while len(tour) < n:
        i = tour[-1]
        neighbours = [(j, dis_matrix[i, j]) for j in nodes if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)
    tour.append(depot)

    route2End = np.zeros((n, 2))
    route2End[0, 0] = tour[-2]
    route2End[0, 1] = tour[1]
    for i in range(1, n):
        route2End[tour[i], 0] = tour[i - 1]
        route2End[tour[i], 1] = tour[i + 1]
    return route2End


def two_opt(tour, i, j):
    if i == j:
        return tour
    a = tour[i, 0]
    b = tour[j, 0]
    tour[i, 0] = tour[i, 1]
    tour[i, 1] = j
    tour[j, 0] = i
    tour[a, 1] = b
    tour[b, 1] = tour[b, 0]
    tour[b, 0] = a
    c = tour[b, 1]
    while tour[c, 1] != j:
        d = tour[c, 0]
        tour[c, 0] = tour[c, 1]
        tour[c, 1] = d
        c = d
    return tour


def two_opt_cost(tour, D, i, j):
    if i == j:
        return 0
    a = tour[i, 0]
    b = tour[j, 0]
    delta = D[a, b] + D[i, j] - D[a, i] - D[b, j]
    return delta


def two_opt_a2a(tour, D, N, first_improvement=False, set_delta=0):
    best_move = None
    best_delta = set_delta

    for i in range(0, len(tour) - 1):
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


def two_opt_o2a_all(tour, D, N, i):
    best_delta = 0
    for j in N[i]:
        if i in tour[j] or j in tour[i]:
            continue
        delta = two_opt_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            tour = two_opt(tour, i, j)
    return best_delta, tour


def relocate(tour, i, j):
    a = tour[i, 0]
    b = tour[i, 1]
    tour[a, 1] = b
    tour[b, 0] = a

    d = tour[j, 1]
    tour[d, 0] = i
    tour[i, 0] = j
    tour[i, 1] = d
    tour[j, 1] = i
    return tour


def relocate_cost(tour, D, i, j):
    if i == j:
        return 0
    a = tour[i, 0]
    b = i
    c = tour[i, 1]
    d = j
    e = tour[j, 1]
    delta = - D[a, b] - D[b, c] + D[a, c] - D[d, e] + D[d, b] + D[b, e]
    return delta


def relocate_o2a_all(tour, D, N, i):
    best_delta = 0
    for j in N[i]:
        if tour[j, 1] == i:  # e.g. relocate 2 -> 3 == relocate 3 -> 2
            continue
        delta = relocate_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            tour = relocate(tour, i, j)
    return best_delta, tour


def relocate_a2a(tour, D, N, first_improvement=False, set_delta=0):
    best_move = None
    best_delta = set_delta

    for i in range(0, len(tour) - 1):
        for j in N[i]:
            if tour[j, 1] == i:  # e.g. relocate 2 -> 3 == relocate 3 -> 2
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


def route2tour(route):
    s = 0
    tour = []
    for _ in range(len(route)):
        tour.append(route[s, 1])
        s = route[s, 1]
    return tour


def local_search(init_tour, init_cost, D, N, first_improvement=False):
    cur_route, cur_cost = init_tour, init_cost
    improved = True
    while improved:
        improved = False

        delta, new_tour = two_opt_a2a(cur_route, D, N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour

        delta, new_tour = relocate_a2a(cur_route, D, N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour

    return cur_route, cur_cost


def guided_local_search(edge_weight, nearest_indices, init_tour, init_cost,
                        t_lim, ite_max, perturbation_moves, update_fn,
                        first_improvement=False):
    # Fixed seed so the search is reproducible across runs.
    random.seed(2024)

    cur_route, cur_cost = local_search(init_tour, init_cost, edge_weight,
                                       nearest_indices, first_improvement)
    best_route, best_cost = cur_route, cur_cost

    length = len(edge_weight[0])
    iter_i = 0
    edge_penalty = np.zeros((length, length))

    while iter_i < ite_max and time.time() < t_lim:

        for _move in range(perturbation_moves):

            cur_tour, best_tour = route2tour(cur_route), route2tour(best_route)

            # LLM-designed heuristic augments the distance matrix.
            edge_weight_guided = update_fn(edge_weight, np.array(cur_tour), edge_penalty)
            edge_weight_guided = np.asmatrix(edge_weight_guided)

            edge_weight_gap = edge_weight_guided - edge_weight

            # Penalise the 5 most-augmented edges and re-optimise around them.
            for _topid in range(5):
                max_indices = np.argmin(-edge_weight_gap, axis=None)
                rows, columns = np.unravel_index(max_indices, edge_weight_gap.shape)

                edge_penalty[rows, columns] += 1
                edge_penalty[columns, rows] += 1
                edge_weight_gap[rows, columns] = 0
                edge_weight_gap[columns, rows] = 0

                for node in [rows, columns]:
                    delta, new_route = two_opt_o2a_all(cur_route, edge_weight_guided,
                                                       nearest_indices, node)
                    if delta < 0:
                        cur_cost = tour_cost_2End(edge_weight, new_route)
                        cur_route = new_route
                    delta, new_route = relocate_o2a_all(cur_route, edge_weight_guided,
                                                        nearest_indices, node)
                    if delta < 0:
                        cur_cost = tour_cost_2End(edge_weight, new_route)
                        cur_route = new_route

        cur_route, cur_cost = local_search(cur_route, cur_cost, edge_weight,
                                           nearest_indices, first_improvement)
        cur_cost = tour_cost_2End(edge_weight, cur_route)

        if cur_cost < best_cost:
            best_route, best_cost = cur_route, cur_cost
        iter_i += 1

        if iter_i % 50 == 0:
            cur_route, cur_cost = best_route, best_cost

    return best_route, best_cost, iter_i


def solve_instance(opt_cost, dis_matrix, time_limit, ite_max,
                   perturbation_moves, update_fn):
    """Run GLS on one instance and return the optimality gap (%)."""
    t = time.time()
    try:
        init_tour = nearest_neighbor_2End(dis_matrix, 0).astype(int)
        init_cost = tour_cost_2End(dis_matrix, init_tour)
        nb = 100
        nearest_indices = np.argsort(dis_matrix, axis=1)[:, 1:nb + 1].astype(int)

        best_tour, best_cost, iter_i = guided_local_search(
            dis_matrix, nearest_indices, init_tour, init_cost,
            t + time_limit, ite_max, perturbation_moves, update_fn,
            first_improvement=False)

        gap = (best_cost / opt_cost - 1) * 100
    except Exception:
        gap = 1E10

    return gap


class TSPGLS(BaseProblem):
    """TSP Guided Local Search.

    The LLM designs update_edge_distance, which modifies the distance matrix
    at each GLS iteration to guide the search away from current local optima.

    GLS loop (per iteration):
      1. Full 2-opt + relocate local search (on candidate neighbour lists).
      2. Call update_edge_distance to get augmented distances.
      3. Penalise the 5 most-augmented edges and run targeted 2-opt + relocate
         moves around their endpoints on the augmented distances.
      4. Record the tour cost on the *original* distances; keep the best.
      5. Every 50 iterations restart from the best-so-far tour.

    Fitness: average optimality gap (%) versus the known optimal tours across
    all training instances (lower = better).
    """

    template_program = '''
def update_edge_distance(edge_distance: np.ndarray, local_opt_tour: np.ndarray,
                          edge_n_used: np.ndarray) -> np.ndarray:
    """Modify the edge distance matrix to escape the current local optimum.

    Args:
        edge_distance:  n*n symmetric matrix of original pairwise distances
        local_opt_tour: length-n array of node indices forming the current
                        local-optimal tour
        edge_n_used:    n*n symmetric matrix counting how many times each
                        edge has been penalised in previous iterations
    Returns:
        updated_edge_distance: modified n*n distance matrix to be used in
                               the next local search
    """
    return edge_distance.copy()
'''

    task_description = (
        "Given a local-optimal TSP tour and the original edge distance matrix, "
        "design a novel strategy to update the distance matrix so that a guided "
        "local search escapes the current local optimum. "
        "The modified distances steer the local search towards unexplored regions "
        "of the solution space. The goal is to minimise the final tour length."
    )

    def __init__(self, n_inst_eva: int = 3, time_limit: float = 10.0,
                 ite_max: int = 1000, perturbation_moves: int = 1,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_inst_eva = n_inst_eva
        self.time_limit = time_limit
        self.ite_max = ite_max
        self.perturbation_moves = perturbation_moves
        self.coords, self.instances, self.opt_costs = GetData(n_inst_eva).load_instances()

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        gaps = np.zeros(self.n_inst_eva)
        for i in range(self.n_inst_eva):
            gaps[i] = solve_instance(
                self.opt_costs[i], self.instances[i],
                self.time_limit, self.ite_max,
                self.perturbation_moves, callable_func)
        return float(np.mean(gaps))
