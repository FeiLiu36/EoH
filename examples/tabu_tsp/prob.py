# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


def _tour_cost(tour: list, dist: np.ndarray) -> float:
    return float(sum(dist[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


def _nearest_neighbour(dist: np.ndarray, start: int = 0) -> list:
    n = len(dist)
    visited = [False] * n
    visited[start] = True
    tour = [start]
    for _ in range(n - 1):
        cur = tour[-1]
        nxt = min((j for j in range(n) if not visited[j]), key=lambda j: dist[cur, j])
        tour.append(nxt)
        visited[nxt] = True
    tour.append(start)
    return tour


class TabuTSP(BaseProblem):
    """EoH task: automatically design the move-scoring function for Tabu Search on TSP.

    The harness runs Tabu Search with 2-opt moves on random Euclidean TSP instances.
    Tours are initialised with nearest-neighbour and then improved by the Tabu search.

    The LLM designs `score_moves`, called once per iteration with ALL candidate
    2-opt moves simultaneously as NumPy arrays. The move with the highest finite
    score is executed. The function can encode aspiration criteria, exploration
    bonuses, diversification strategies, and adaptive tabu overrides.

    Fitness: mean tour cost across all training instances and seeds (lower is better).
    """

    template_program = '''
def score_moves(delta_costs: np.ndarray, is_tabu_mask: np.ndarray,
                best_cost: float, current_cost: float, tabu_ages: np.ndarray,
                iteration: int, max_iterations: int) -> np.ndarray:
    """Score all candidate 2-opt moves for Tabu Search on TSP.

    Called once per iteration with all M candidate moves simultaneously.
    The harness executes the move with the highest finite score.

    Args:
        delta_costs:    float array shape (M,) — change in tour cost for each
                        2-opt move (negative = improvement, positive = worsening)
        is_tabu_mask:   bool array shape (M,) — True if the move is currently tabu
        best_cost:      best tour cost found so far in this run
        current_cost:   current tour cost before any move this iteration
        tabu_ages:      int array shape (M,) — 0 for non-tabu moves; number of
                        iterations since the move was added to the tabu list otherwise
        iteration:      current iteration index (0-based)
        max_iterations: total Tabu Search iterations planned
    Returns:
        scores: float array shape (M,) — move with highest finite score is selected.
                Return -np.inf for any move to forbid it entirely.
    """
    scores = np.full(len(delta_costs), -np.inf)
    non_tabu = ~is_tabu_mask
    scores[non_tabu] = -delta_costs[non_tabu]
    # Classic aspiration: allow tabu move if it would beat the global best
    aspiration = is_tabu_mask & (current_cost + delta_costs < best_cost)
    scores[aspiration] = -delta_costs[aspiration] + 1e6
    return scores
'''

    task_description = (
        "Design a novel move-scoring function for Tabu Search applied to the "
        "Travelling Salesman Problem (TSP) with 2-opt neighbourhood moves. "
        "At each iteration the harness provides all candidate 2-opt moves as NumPy "
        "arrays and executes the move with the highest score returned by your function. "
        "Input arrays (all shape (M,)): "
        "delta_costs — change in tour length (negative = better); "
        "is_tabu_mask — whether each move is currently forbidden by the tabu list; "
        "tabu_ages — how many iterations ago each tabu move was added (0 if not tabu); "
        "plus scalars: best_cost (best tour so far), current_cost, iteration, max_iterations. "
        "The classic baseline selects the best non-tabu move and uses a simple aspiration "
        "criterion (override tabu if the move reaches a new global best). "
        "Alternatives include: progress-adaptive aspiration thresholds, "
        "diversification bonuses that favour long-tabu moves late in the search, "
        "probabilistic acceptance of worsening moves in early iterations, "
        "or hybrid rules that dynamically tighten/loosen the aspiration criterion. "
        "The goal is to minimise the mean final tour cost across random Euclidean TSP instances."
    )

    def __init__(self, n_nodes: int = 20, n_instances: int = 5,
                 n_iter: int = 200, tabu_tenure: int = 7, n_runs: int = 3,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_nodes = n_nodes
        self.n_iter = n_iter
        self.tabu_tenure = tabu_tenure
        self.n_runs = n_runs
        self.instances = GetData.generate_instances(n_instances, n_nodes, seed=42)
        # Precompute all 2-opt move index pairs for n_nodes
        self._move_pairs = [
            (i, j)
            for i in range(1, n_nodes - 1)
            for j in range(i + 1, n_nodes)
        ]

    def _run_tabu(self, dist: np.ndarray, score_fn, seed: int) -> float:
        """Run one Tabu Search trial; return the best tour cost found."""
        rng = np.random.RandomState(seed)
        n = len(dist)
        start = int(rng.randint(n))
        tour = _nearest_neighbour(dist, start)
        current_cost = _tour_cost(tour, dist)
        best_cost = current_cost

        # tabu_list: edge-pair key → iteration when it expires
        tabu_expiry: dict[tuple, int] = {}
        move_pairs = self._move_pairs
        M = len(move_pairs)

        for iteration in range(self.n_iter):
            # Build move arrays
            delta_costs = np.empty(M, dtype=float)
            is_tabu_mask = np.zeros(M, dtype=bool)
            tabu_ages = np.zeros(M, dtype=int)
            move_keys = []

            for k, (i, j) in enumerate(move_pairs):
                delta = (
                    dist[tour[i - 1], tour[j]]
                    + dist[tour[i], tour[j + 1]]
                    - dist[tour[i - 1], tour[i]]
                    - dist[tour[j], tour[j + 1]]
                )
                delta_costs[k] = delta

                # Move key: the new edge (tour[i-1], tour[j]) added by this swap
                u, v = tour[i - 1], tour[j]
                key = (min(u, v), max(u, v))
                move_keys.append(key)
                expiry = tabu_expiry.get(key, 0)
                if iteration < expiry:
                    is_tabu_mask[k] = True
                    tabu_ages[k] = iteration - (expiry - self.tabu_tenure)

            # Score all moves
            try:
                scores = score_fn(
                    delta_costs, is_tabu_mask, float(best_cost),
                    float(current_cost), tabu_ages,
                    int(iteration), int(self.n_iter),
                )
                scores = np.asarray(scores, dtype=float)
            except Exception:
                scores = np.full(M, -np.inf)

            # Select best move
            finite_mask = np.isfinite(scores)
            if not finite_mask.any():
                continue

            best_k = int(np.argmax(np.where(finite_mask, scores, -np.inf)))
            i, j = move_pairs[best_k]

            # Apply 2-opt reversal
            tour[i:j + 1] = tour[i:j + 1][::-1]
            current_cost += delta_costs[best_k]

            # Update tabu list with the new edge added by this move
            tabu_expiry[move_keys[best_k]] = iteration + self.tabu_tenure

            if current_cost < best_cost:
                best_cost = current_cost

        return float(best_cost)

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        costs = []
        for instance in self.instances:
            dist = instance['dist']
            for seed in range(self.n_runs):
                costs.append(self._run_tabu(dist, callable_func, seed))
        return float(np.mean(costs))
