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


# ── pure-Python GLS primitives (no numba) ─────────────────────────────────────

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


class TSPGLS(BaseProblem):
    """TSP Guided Local Search.

    The LLM designs update_edge_distance, which modifies the distance matrix
    at each GLS iteration to guide the search away from current local optima.

    GLS loop (per iteration):
      1. Call update_edge_distance to get augmented distances.
      2. Identify the 5 most-augmented edges and increment their penalty counts.
      3. Run 2-opt local search on the augmented distances.
      4. Record the tour cost on the *original* distances.

    Fitness: average final tour length across all training instances (lower = better).
    """

    template_program = '''
def update_edge_distance(edge_distance: np.ndarray, local_opt_tour: np.ndarray,
                          edge_n_used: np.ndarray) -> np.ndarray:
    """Modify the edge distance matrix to escape the current local optimum.

    Args:
        edge_distance:  n*n symmetric matrix of original pairwise distances
        local_opt_tour: length-n array of node indices forming the current
                        local-optimal tour (excludes the closing return edge)
        edge_n_used:    n*n symmetric matrix counting how many times each
                        edge has been penalised in previous iterations
    Returns*
        updated_edge_distance: modified n*n distance matrix to be used in
                               the next 2-opt local search
    """
    return edge_distance.copy()
'''

    task_description = (
        "Given a local-optimal TSP tour and the original edge distance matrix, "
        "design a novel strategy to update the distance matrix so that a guided "
        "local search escapes the current local optimum. "
        "The modified distances steer 2-opt towards unexplored regions of the "
        "solution space. The goal is to minimise the final tour length."
    )

    def __init__(self, n_nodes: int = 20, n_instance: int = 3,
                 iter_max: int = 100, time_max: float = 5.0,
                 timeout: int = 40, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_nodes = n_nodes
        self.n_instance = n_instance
        self.iter_max = iter_max
        self.time_max = time_max
        self.instance_data = GetData(n_instance, n_nodes).generate_instances()

    def _gls(self, dist, update_fn):
        n = len(dist)
        tour = _nearest_neighbour(dist)
        tour = _two_opt(tour, dist)
        best_cost = _tour_cost(tour, dist)
        best_tour = tour[:]
        edge_n_used = np.zeros((n, n))

        t_end = time.time() + self.time_max
        for _ in range(self.iter_max):
            if time.time() > t_end:
                break

            # Augment distances with the LLM-designed heuristic
            aug = update_fn(
                dist.copy(),
                np.array(tour[:-1], dtype=int),
                edge_n_used.copy(),
            )
            aug = np.asarray(aug, dtype=float)
            aug = (aug + aug.T) / 2        # enforce symmetry
            np.maximum(aug, 0, out=aug)    # enforce non-negativity

            # Penalise the 5 edges with the highest augmentation
            gain = aug - dist
            np.fill_diagonal(gain, -np.inf)
            for _ in range(5):
                u, v = np.unravel_index(int(np.argmax(gain)), gain.shape)
                edge_n_used[u, v] += 1
                edge_n_used[v, u] += 1
                gain[u, v] = gain[v, u] = -np.inf

            # 2-opt on the augmented distances, evaluate on original
            tour = _two_opt(best_tour[:], aug)
            cost = _tour_cost(tour, dist)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour[:]

        return best_cost

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        costs = []
        for _, dist in self.instance_data:
            costs.append(self._gls(dist, callable_func))
        return float(np.mean(costs))
