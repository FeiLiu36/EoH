import importlib
import time
import numpy as np

# Re-use the pure-Python GLS primitives from prob.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prob import _tour_cost, _nearest_neighbour, _two_opt


class Evaluation:
    def __init__(self, dataset, n_test, iter_max=200, time_max=10.0):
        self.instance_data = dataset[:n_test]
        self.iter_max = iter_max
        self.time_max = time_max

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
            aug = update_fn(dist.copy(), np.array(tour[:-1], dtype=int), edge_n_used.copy())
            aug = np.asarray(aug, dtype=float)
            aug = (aug + aug.T) / 2
            np.maximum(aug, 0, out=aug)
            gain = aug - dist
            np.fill_diagonal(gain, -np.inf)
            for _ in range(5):
                u, v = np.unravel_index(int(np.argmax(gain)), gain.shape)
                edge_n_used[u, v] += 1
                edge_n_used[v, u] += 1
                gain[u, v] = gain[v, u] = -np.inf
            tour = _two_opt(best_tour[:], aug)
            cost = _tour_cost(tour, dist)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour[:]
        return best_cost

    def evaluate(self):
        mod = importlib.reload(importlib.import_module("heuristic"))
        costs = [self._gls(dist, mod.update_edge_distance)
                 for _, dist in self.instance_data]
        return float(np.mean(costs))
