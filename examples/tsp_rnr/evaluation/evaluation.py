import importlib
import time
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prob import _tour_cost, _nearest_neighbour, _two_opt, _cheapest_insertion


class Evaluation:
    def __init__(self, dataset, n_test, n_destroy=None,
                 iter_max=200, time_max=10.0):
        self.instance_data = dataset[:n_test]
        self.iter_max = iter_max
        self.time_max = time_max
        # n_destroy is set per instance based on tour length if not fixed
        self._n_destroy = n_destroy

    def _rnr(self, dist, destroy_fn):
        n = len(dist)
        n_destroy = self._n_destroy if self._n_destroy is not None else max(2, n // 5)

        tour = _nearest_neighbour(dist)
        tour = _two_opt(tour, dist)
        best_cost = _tour_cost(tour, dist)
        best_tour = tour[:]

        t_end = time.time() + self.time_max
        for _ in range(self.iter_max):
            if time.time() > t_end:
                break

            open_tour = np.array(best_tour[:-1], dtype=int)

            nodes_to_remove = destroy_fn(open_tour.copy(), dist.copy(), n_destroy)
            nodes_to_remove = np.asarray(nodes_to_remove, dtype=int).flatten()
            if len(nodes_to_remove) < n_destroy:
                continue
            nodes_to_remove = nodes_to_remove[:n_destroy]
            if not all(0 <= v < n for v in nodes_to_remove):
                continue

            removed_set = set(nodes_to_remove.tolist())
            partial = [v for v in open_tour if v not in removed_set]
            if len(partial) < 2:
                continue
            new_open = _cheapest_insertion(partial, nodes_to_remove.tolist(), dist)
            new_tour = new_open + [new_open[0]]

            new_tour = _two_opt(new_tour, dist)
            cost = _tour_cost(new_tour, dist)
            if cost < best_cost:
                best_cost = cost
                best_tour = new_tour[:]

        return best_cost

    def evaluate(self):
        mod = importlib.reload(importlib.import_module("heuristic"))
        costs = [self._rnr(dist, mod.destroy_nodes)
                 for _, dist in self.instance_data]
        return float(np.mean(costs))
