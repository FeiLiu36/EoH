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
from get_instance import GetData, hypervolume_2d


class MoBbobMetaheuristic(BaseProblem):
    """EoH task: design a multi-objective metaheuristic that approximates the Pareto front.

    The LLM designs a Metaheuristic class that runs the full multi-objective search
    internally and returns its Pareto front approximation as a decision-space array.

    Interface contract:
        __init__(self, func, dim, bounds, budget, n_obj)
            func   : callable f(x: np.ndarray) -> np.ndarray shape (n_obj,);
                     lower is better for each objective
            dim    : int, number of decision variables
            bounds : np.ndarray shape (2, dim); bounds[0]=lower, bounds[1]=upper
            budget : int, maximum number of func evaluations allowed
            n_obj  : int, number of objectives (always 2 for this task)
        solve(self) -> np.ndarray shape (k, dim)
            return : 2-D array of k solutions approximating the Pareto front
                     k can be any positive integer

    Fitness is mean negative hypervolume across all benchmark instances and seeds
    (lower is better, meaning the optimiser should maximise hypervolume).
    The harness evaluates all returned solutions to compute hypervolume, so
    only the quality of the returned front matters — internal bookkeeping is free.
    """

    template_program = '''
class Metaheuristic:

    def __init__(self, func, dim, bounds, budget, n_obj):
        # func   : callable f(x: np.ndarray) -> np.ndarray shape (n_obj,)
        # bounds : np.ndarray shape (2, dim); bounds[0]=lower, bounds[1]=upper
        # budget : int, maximum func evaluations allowed
        # n_obj  : int, number of objectives
        self.func   = func
        self.dim    = dim
        self.bounds = bounds
        self.budget = budget
        self.n_obj  = n_obj

    def solve(self):
        lo, hi = self.bounds[0], self.bounds[1]
        archive_x, archive_f = [], []

        for _ in range(self.budget):
            x = lo + (hi - lo) * np.random.rand(self.dim)
            f = self.func(x)
            archive_x.append(x.copy())
            archive_f.append(f.copy())

        X = np.array(archive_x)
        F = np.array(archive_f)

        # Filter to non-dominated solutions
        nd = np.ones(len(F), dtype=bool)
        for i in range(len(F)):
            dominated_by = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            dominated_by[i] = False
            if dominated_by.any():
                nd[i] = False
        return X[nd]
'''

    task_description = (
        "Design a multi-objective black-box metaheuristic as a Python class named Metaheuristic. "
        "__init__(self, func, dim, bounds, budget, n_obj) receives all inputs: "
        "func is a callable f(x: np.ndarray) -> np.ndarray of shape (n_obj,) returning objective "
        "values (lower is better for each objective); "
        "dim is the number of decision variables (int); "
        "bounds is a numpy array of shape (2, dim): bounds[0] contains per-dimension lower bounds "
        "and bounds[1] contains per-dimension upper bounds (bounds may differ across dimensions); "
        "budget is the total number of func evaluations allowed (int); "
        "n_obj is the number of objectives (int, always 2 for this task). "
        "The class must implement solve(self) -> np.ndarray of shape (k, dim), which runs the "
        "full multi-objective search and returns k solutions approximating the Pareto front. "
        "The class may use any internal data structures and algorithms. "
        "Fitness is mean negative hypervolume (lower is better) across four ZDT benchmark instances "
        "(ZDT1, ZDT2, ZDT3, ZDT4) in 10 dimensions — larger hypervolume means a better Pareto "
        "front approximation."
    )

    def __init__(self, dim: int = 10, budget: int = 5000, n_instances: int = 4,
                 n_runs: int = 5, timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.dim = dim
        self.budget = budget
        self.n_runs = n_runs
        self.instances = GetData().get_instances(dim=dim, n_instances=n_instances)

    def _run_one(self, instance: dict, MetaheuristicClass, seed: int) -> float:
        """Run one solve trial; return hypervolume of the returned Pareto front."""
        np.random.seed(seed)
        func   = instance['func']
        lo, hi = instance['bounds']          # per-dimension arrays
        bounds = np.array([lo, hi])
        ref_pt = instance['ref_pt']
        n_obj  = instance['n_obj']

        solver = MetaheuristicClass(func, self.dim, bounds, self.budget, n_obj)
        X_front = solver.solve()

        X_front = np.clip(np.asarray(X_front, dtype=float).reshape(-1, self.dim), lo, hi)
        F_front = np.array([func(x) for x in X_front])
        return hypervolume_2d(F_front, ref_pt)

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        hvs = []
        for instance in self.instances:
            for seed in range(self.n_runs):
                hvs.append(self._run_one(instance, callable_func, seed))
        # Negative mean HV: EoH minimises fitness, so this maximises HV
        return float(-np.mean(hvs))
