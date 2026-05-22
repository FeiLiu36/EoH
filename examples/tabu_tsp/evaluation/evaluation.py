import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


class Evaluation:
    """Post-hoc evaluator for Tabu Search move-scoring functions on TSP.

    Uses larger instances, more seeds, and more iterations than the training evaluator
    in prob.py. Tests on 20-node and 30-node random Euclidean instances.
    """

    CONFIGS = [
        {'n_nodes': 20, 'n_instances': 10, 'seed': 100},
        {'n_nodes': 30, 'n_instances': 10, 'seed': 200},
    ]

    def __init__(self, n_iter: int = 500, tabu_tenure: int = 7, n_runs: int = 10):
        self.n_iter = n_iter
        self.tabu_tenure = tabu_tenure
        self.n_runs = n_runs
        self.groups = []
        for cfg in self.CONFIGS:
            instances = GetData.generate_instances(
                cfg['n_instances'], cfg['n_nodes'], seed=cfg['seed']
            )
            move_pairs = [
                (i, j)
                for i in range(1, cfg['n_nodes'] - 1)
                for j in range(i + 1, cfg['n_nodes'])
            ]
            self.groups.append({
                'n_nodes': cfg['n_nodes'],
                'instances': instances,
                'move_pairs': move_pairs,
            })

    def _run_tabu(self, dist: np.ndarray, move_pairs: list,
                  score_fn, seed: int) -> float:
        rng = np.random.RandomState(seed)
        n = len(dist)
        start = int(rng.randint(n))
        tour = _nearest_neighbour(dist, start)
        current_cost = _tour_cost(tour, dist)
        best_cost = current_cost
        tabu_expiry: dict[tuple, int] = {}
        M = len(move_pairs)

        for iteration in range(self.n_iter):
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
                u, v = tour[i - 1], tour[j]
                key = (min(u, v), max(u, v))
                move_keys.append(key)
                expiry = tabu_expiry.get(key, 0)
                if iteration < expiry:
                    is_tabu_mask[k] = True
                    tabu_ages[k] = iteration - (expiry - self.tabu_tenure)

            try:
                scores = score_fn(
                    delta_costs, is_tabu_mask, float(best_cost),
                    float(current_cost), tabu_ages,
                    int(iteration), int(self.n_iter),
                )
                scores = np.asarray(scores, dtype=float)
            except Exception:
                scores = np.full(M, -np.inf)

            finite_mask = np.isfinite(scores)
            if not finite_mask.any():
                continue

            best_k = int(np.argmax(np.where(finite_mask, scores, -np.inf)))
            i, j = move_pairs[best_k]
            tour[i:j + 1] = tour[i:j + 1][::-1]
            current_cost += delta_costs[best_k]
            tabu_expiry[move_keys[best_k]] = iteration + self.tabu_tenure

            if current_cost < best_cost:
                best_cost = current_cost

        return float(best_cost)

    def evaluate(self, score_fn) -> list[dict]:
        """Evaluate score_fn on the full benchmark suite.

        Returns a list of result dicts with keys: n_nodes, mean, std, mean_per_node.
        """
        results = []
        for group in self.groups:
            n_nodes = group['n_nodes']
            move_pairs = group['move_pairs']
            all_costs = []
            for instance in group['instances']:
                dist = instance['dist']
                for seed in range(self.n_runs):
                    all_costs.append(self._run_tabu(dist, move_pairs, score_fn, seed))
            results.append({
                'n_nodes':       n_nodes,
                'mean':          float(np.mean(all_costs)),
                'std':           float(np.std(all_costs)),
                'mean_per_node': float(np.mean(all_costs) / n_nodes),
            })
        return results
