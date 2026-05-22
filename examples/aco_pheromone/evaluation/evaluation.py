import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData


class Evaluation:
    """Post-hoc evaluator for ACO pheromone update rules.

    Uses larger instances, more ants, more iterations, and more seeds than
    the training evaluator in prob.py, giving a thorough performance picture.
    Results are reported per instance and as an overall average.
    """

    def __init__(self, n_cities: int = 50, n_instance: int = 5,
                 n_ants: int = 25, iter_max: int = 200,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1,
                 n_runs: int = 10):
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

    def evaluate(self, update_fn) -> list[dict]:
        """Run update_fn on all instances and return per-instance result dicts.

        Each dict has keys: instance_id, mean, std.
        """
        results = []
        for idx, (_, dist) in enumerate(self.instance_data):
            run_costs = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                run_costs.append(self._run_aco(dist, update_fn))
            results.append({
                'instance_id': idx,
                'mean': float(np.mean(run_costs)),
                'std':  float(np.std(run_costs)),
            })
        return results
