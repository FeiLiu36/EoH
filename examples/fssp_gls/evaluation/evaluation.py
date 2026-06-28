import importlib
import sys
import os

import numpy as np

# Re-use the FSSP/GLS engine from prob.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prob import gls


class Evaluation:
    """Evaluate an evolved heuristic on a set of flow-shop instances.

    Reports the average final makespan (lower = better).
    """

    def __init__(self, instances, time_max=30.0, iter_max=1000):
        # instances: list of (tasks_val, machines_val, tasks) tuples
        self.instances = instances
        self.time_max = time_max
        self.iter_max = iter_max

    def evaluate(self):
        mod = importlib.reload(importlib.import_module("heuristic"))
        cmax = [gls(tv, t, mv, self.time_max, self.iter_max, mod.get_matrix_and_jobs)
                for tv, mv, t in self.instances]
        return float(np.mean(cmax))
