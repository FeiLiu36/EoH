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


class ACOPheromone(BaseProblem):
    """EoH task: automatically design the pheromone update rule for ACO on TSP.

    The LLM designs `update_pheromone`. The harness provides a fixed
    probabilistic ant construction procedure (standard ACO transition rule)
    and evaluates the average best tour length found across TSP instances.
    Lower fitness is better.
    """

    template_program = '''
def update_pheromone(pheromone: np.ndarray, ant_tours: list, tour_costs: np.ndarray,
                     best_tour: np.ndarray, best_cost: float,
                     rho: float, iteration: int, max_iterations: int) -> np.ndarray:
    """Update the pheromone matrix after one ACO iteration (Ant System baseline).

    Args:
        pheromone:      (n, n) symmetric matrix of current pheromone levels
        ant_tours:      list of m arrays, each of shape (n,) – city visit sequences
                        (ant_tours[k][i] is the i-th city visited by ant k)
        tour_costs:     (m,) tour lengths for each ant (lower = better)
        best_tour:      (n,) best tour found so far across all iterations
        best_cost:      length of best_tour
        rho:            evaporation rate in (0, 1)
        iteration:      current iteration index (0-based)
        max_iterations: total number of ACO iterations planned
    Returns:
        updated pheromone matrix of shape (n, n), values should be non-negative
    """
    n = pheromone.shape[0]
    pheromone = (1.0 - rho) * pheromone
    for tour, cost in zip(ant_tours, tour_costs):
        delta = 1.0 / cost
        for i in range(n):
            u, v = int(tour[i]), int(tour[(i + 1) % n])
            pheromone[u, v] += delta
            pheromone[v, u] += delta
    return pheromone
'''

    task_description = (
        "Design a novel pheromone update rule for the Ant Colony Optimisation (ACO) "
        "algorithm applied to the Travelling Salesman Problem (TSP). "
        "After each iteration, ants construct tours and the pheromone matrix is "
        "updated to reinforce promising edges and evaporate others. "
        "Classic strategies include Ant System (all ants deposit proportional to 1/cost), "
        "Elitist AS (best-so-far ant gets bonus deposit), "
        "MMAS (only best ant deposits, pheromone clamped to [tau_min, tau_max]), and "
        "Rank-based AS (top-w ants deposit with linearly decreasing weights). "
        "You may design adaptive rules that change behaviour over iterations. "
        "The goal is to minimise the average best tour length found across TSP instances."
    )

    def __init__(self, n_cities: int = 20, n_instance: int = 3,
                 n_ants: int = 20, iter_max: int = 100,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1,
                 n_runs: int = 3, timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_cities = n_cities
        self.n_ants = n_ants
        self.iter_max = iter_max
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_runs = n_runs
        self.instance_data = GetData(n_instance, n_cities).generate_instances()

    def _tour_cost(self, tour: np.ndarray, dist: np.ndarray) -> float:
        n = len(tour)
        return float(sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n)))

    def _construct_tour(self, pheromone: np.ndarray, eta: np.ndarray) -> np.ndarray:
        n = pheromone.shape[0]
        start = np.random.randint(n)
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour = [start]
        for _ in range(n - 1):
            current = tour[-1]
            attract = (pheromone[current] ** self.alpha) * (eta[current] ** self.beta)
            attract[visited] = 0.0
            total = attract.sum()
            if total < 1e-300:
                attract = (~visited).astype(float)
                total = attract.sum()
            probs = attract / total
            nxt = int(np.random.choice(n, p=probs))
            tour.append(nxt)
            visited[nxt] = True
        return np.array(tour, dtype=int)

    def _run_aco(self, dist: np.ndarray, update_fn) -> float:
        n = len(dist)
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = np.where(dist > 0, 1.0 / dist, 0.0)
        np.fill_diagonal(eta, 0.0)

        pheromone = np.ones((n, n))
        best_tour = None
        best_cost = np.inf

        for it in range(self.iter_max):
            ant_tours, tour_costs = [], []
            for _ in range(self.n_ants):
                tour = self._construct_tour(pheromone, eta)
                cost = self._tour_cost(tour, dist)
                ant_tours.append(tour)
                tour_costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour.copy()

            pheromone = update_fn(
                pheromone.copy(),
                ant_tours,
                np.array(tour_costs, dtype=float),
                best_tour.copy(),
                float(best_cost),
                self.rho,
                it,
                self.iter_max,
            )
            pheromone = np.maximum(pheromone, 1e-10)

        return float(best_cost)

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        costs = []
        for _, dist in self.instance_data:
            run_costs = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                run_costs.append(self._run_aco(dist, callable_func))
            costs.append(float(np.mean(run_costs)))
        return float(np.mean(costs))
