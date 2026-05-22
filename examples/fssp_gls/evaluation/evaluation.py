import importlib
import time
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prob import _makespan, _neh, _local_search, _local_search_targeted


class Evaluation:
    def __init__(self, dataset, n_test, n_jobs, n_machines,
                 iter_max=200, time_max=15.0):
        self.instance_data = dataset[:n_test]
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.iter_max = iter_max
        self.time_max = time_max

    def _gls(self, tasks, heuristic):
        seq, cmax = _neh(tasks)
        best_seq, best_cmax = seq[:], cmax
        t_end = time.time() + self.time_max

        for _ in range(self.iter_max):
            if time.time() > t_end:
                break
            seq, cmax = _local_search(seq, tasks, t_end)
            if cmax < best_cmax:
                best_seq, best_cmax = seq[:], cmax

            result = heuristic(seq[:], tasks.copy(), self.n_machines, self.n_jobs)
            new_matrix, perturb_jobs = result
            new_matrix = np.asarray(new_matrix, dtype=float)
            if new_matrix.shape != tasks.shape:
                continue
            perturb_jobs = [int(j) for j in list(perturb_jobs)[:5]
                            if 0 <= int(j) < self.n_jobs]
            if len(perturb_jobs) < 2:
                continue
            seq, _ = _local_search_targeted(seq, new_matrix, perturb_jobs)
            cmax = _makespan(seq, tasks)
            if cmax < best_cmax:
                best_seq, best_cmax = seq[:], cmax

        return best_cmax

    def evaluate(self):
        mod = importlib.reload(importlib.import_module("heuristic"))
        makespans = [self._gls(tasks, mod.get_matrix_and_jobs)
                     for tasks in self.instance_data]
        return float(np.mean(makespans))
