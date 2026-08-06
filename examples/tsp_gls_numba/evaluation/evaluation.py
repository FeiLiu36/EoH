import importlib
import sys
import os

import numpy as np

# Re-use the pure-Python GLS engine from prob.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prob import solve_instance


class Evaluation:
    """Evaluate an evolved heuristic on a held-out test set of TSP instances.

    Reports the average optimality gap (%) versus the known optimal tours.
    """

    def __init__(self, dataset, n_test, time_limit=10.0, ite_max=1000,
                 perturbation_moves=1):
        self.instances = [np.asarray(d) for d in dataset['distance_matrix'][:n_test]]
        self.opt_costs = list(dataset['cost'][:n_test])
        self.n_test = n_test
        self.time_limit = time_limit
        self.ite_max = ite_max
        self.perturbation_moves = perturbation_moves

    def evaluate(self):
        """Returns (mean_gap, iters_used) where iters_used is the per-instance
        list of GLS iterations actually run (< ite_max means the wall-clock
        time_limit cut the search short rather than ite_max being reached)."""
        mod = importlib.reload(importlib.import_module("heuristic"))
        results = [solve_instance(self.opt_costs[i], self.instances[i],
                                  self.time_limit, self.ite_max,
                                  self.perturbation_moves, mod.update_edge_distance)
                   for i in range(self.n_test)]
        gaps, iters_used = zip(*results)
        return float(np.mean(gaps)), list(iters_used)
