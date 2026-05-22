# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


# ── pure-Python RnR primitives ────────────────────────────────────────────────

def _tour_cost(tour, dist):
    return sum(dist[tour[i], tour[i + 1]] for i in range(len(tour) - 1))


def _nearest_neighbour(dist, start=0):
    n = len(dist)
    visited = [False] * n
    visited[start] = True
    tour = [start]
    for _ in range(n - 1):
        cur = tour[-1]
        nxt = min((j for j in range(n) if not visited[j]),
                  key=lambda j: dist[cur, j])
        tour.append(nxt)
        visited[nxt] = True
    tour.append(start)
    return tour


def _two_opt(tour, dist):
    """Full 2-opt improvement until no improving swap exists."""
    tour = list(tour)
    n = len(tour) - 1
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if (dist[tour[i - 1], tour[j]] + dist[tour[i], tour[j + 1]]
                        < dist[tour[i - 1], tour[i]] + dist[tour[j], tour[j + 1]] - 1e-10):
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def _cheapest_insertion(partial_tour, removed_nodes, dist):
    """Reinsert removed nodes one by one using cheapest insertion."""
    tour = list(partial_tour)
    for node in removed_nodes:
        best_delta = np.inf
        best_pos = 1
        n = len(tour)
        for i in range(n):
            j = (i + 1) % n
            delta = dist[tour[i], node] + dist[node, tour[j]] - dist[tour[i], tour[j]]
            if delta < best_delta:
                best_delta = delta
                best_pos = i + 1
        tour.insert(best_pos, node)
    return tour


class TSPRNR(BaseProblem):
    """TSP Ruin-and-Recreate — destroy operator design.

    The LLM designs the destroy operator, which selects which nodes to remove
    from the current tour during the ruin phase.

    RnR loop (per iteration):
      1. Call destroy_nodes to select n_destroy nodes to remove.
      2. Remove those nodes from the current tour (ruin phase).
      3. Reinsert them greedily via cheapest insertion (recreate phase).
      4. Apply 2-opt local search to the recreated tour.
      5. Accept if the new tour is shorter than the best known.

    Fitness: average best tour length across all training instances (lower = better).
    """

    template_program = '''
def destroy_nodes(current_tour: np.ndarray, distance_matrix: np.ndarray,
                  n_destroy: int) -> np.ndarray:
    """Select nodes to remove from the current tour (ruin phase).

    Args:
        current_tour:    1-D integer array of length n giving the open tour
                         (visit order without the closing return to start)
        distance_matrix: n*n float array of symmetric pairwise distances
        n_destroy:       number of nodes to remove
    Returns:
        nodes_to_remove: integer array of length n_destroy containing the
                         node indices (values in current_tour) to be removed
    """
    return np.random.choice(current_tour, size=n_destroy, replace=False)
'''

    task_description = (
        "Given the current TSP tour and the pairwise distance matrix, "
        "design a novel destroy operator that selects which nodes to remove "
        "during the ruin phase of a ruin-and-recreate algorithm. "
        "Removed nodes are reinserted via cheapest insertion, followed by "
        "2-opt local search. The goal is to minimise the final best tour length."
    )

    def __init__(self, n_nodes: int = 50, n_instance: int = 5,
                 n_destroy: int = None, iter_max: int = 100,
                 time_max: float = 5.0, timeout: int = 40,
                 n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_nodes = n_nodes
        self.n_instance = n_instance
        # default destroy size: ~20% of tour nodes
        self.n_destroy = n_destroy if n_destroy is not None else max(2, n_nodes // 5)
        self.iter_max = iter_max
        self.time_max = time_max
        self.instance_data = GetData(n_instance, n_nodes).generate_instances()

    def _rnr(self, dist, destroy_fn):
        n = len(dist)
        tour = _nearest_neighbour(dist)
        tour = _two_opt(tour, dist)
        best_cost = _tour_cost(tour, dist)
        best_tour = tour[:]

        t_end = time.time() + self.time_max
        for _ in range(self.iter_max):
            if time.time() > t_end:
                break

            open_tour = np.array(best_tour[:-1], dtype=int)

            # Destroy
            nodes_to_remove = destroy_fn(open_tour.copy(), dist.copy(), self.n_destroy)
            nodes_to_remove = np.asarray(nodes_to_remove, dtype=int).flatten()
            if len(nodes_to_remove) < self.n_destroy:
                continue
            nodes_to_remove = nodes_to_remove[:self.n_destroy]
            if not all(0 <= v < n for v in nodes_to_remove):
                continue

            # Recreate
            removed_set = set(nodes_to_remove.tolist())
            partial = [v for v in open_tour if v not in removed_set]
            if len(partial) < 2:
                continue
            new_open = _cheapest_insertion(partial, nodes_to_remove.tolist(), dist)
            new_tour = new_open + [new_open[0]]

            # Local search
            new_tour = _two_opt(new_tour, dist)
            cost = _tour_cost(new_tour, dist)
            if cost < best_cost:
                best_cost = cost
                best_tour = new_tour[:]

        return best_cost

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        costs = []
        for _, dist in self.instance_data:
            costs.append(self._rnr(dist, callable_func))
        return float(np.mean(costs))
