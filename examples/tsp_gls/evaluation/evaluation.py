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
        mod = importlib.reload(importlib.import_module("heuristic"))
        gaps = [solve_instance(self.opt_costs[i], self.instances[i],
                               self.time_limit, self.ite_max,
                               self.perturbation_moves, mod.update_edge_distance)
                for i in range(self.n_test)]
        return float(np.mean(gaps))
