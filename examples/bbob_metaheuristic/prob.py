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


class BbobMetaheuristic(BaseProblem):
    """EoH task: automatically design a complete single-objective black-box metaheuristic.

    The LLM designs a Metaheuristic class that can have any internal structure,
    but must expose a single public method `solve` that runs the search and
    returns the best solution found.

    Interface contract:
        __init__(self, func, dim, bounds, budget)
            func   : callable f(x: np.ndarray) → float, objective (lower is better)
            dim    : int, number of decision variables
            bounds : np.ndarray shape (2, dim); bounds[0]=lower, bounds[1]=upper
            budget : int, maximum number of func evaluations allowed
        solve(self) → x_best (np.ndarray, shape (dim,))
            return : 1-D numpy array of length dim — the best solution found

    Fitness is mean log1p(func(x_best)) across all benchmark instances and
    random seeds — lower is better.
    """

    template_program = '''
class Metaheuristic:

    def __init__(self, func, dim, bounds, budget):
        # func   : callable f(x: np.ndarray) -> float, objective (lower is better)
        # dim    : int, number of decision variables
        # bounds : np.ndarray shape (2, dim); bounds[0]=lower, bounds[1]=upper
        # budget : int, maximum number of func evaluations allowed
        self.func   = func
        self.dim    = dim
        self.bounds = bounds
        self.budget = budget

    def solve(self):
        
        # Implement your solve here. You can define any additional methods or attributes you like.

        return x_best
'''

    task_description = (
        "Design a black-box metaheuristic as a Python class named Metaheuristic. "
        "__init__(self, func, dim, bounds, budget) receives all inputs: "
        "func is a callable f(x: np.ndarray) -> float that evaluates the objective (lower is better); "
        "dim is the number of decision variables (int); "
        "bounds is a numpy array of shape (2, dim): bounds[0] contains lower bounds and bounds[1] contains upper bounds; "
        "budget is the total number of func evaluations allowed (int). "
        "The class must implement solve(self) -> np.ndarray, which runs the search and "
        "returns a 1-D numpy array of length dim representing the best solution found. "
        "The class may have any additional methods or attributes. "
        "Fitness is lower is better."
    )

    def __init__(self, dim: int = 10, budget: int = 100000, n_runs: int = 3,
                 timeout: int = 120, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.dim = dim
        self.budget = budget
        self.n_runs = n_runs
        self.instances = GetData().get_instances(dim=dim)

    def _run_one(self, instance: dict, MetaheuristicClass, seed: int) -> float:
        """Run one solve trial; return the objective value of the returned solution."""
        np.random.seed(seed)
        func = instance['func']
        lo, hi = instance['bounds']
        bounds = np.array([np.full(self.dim, lo), np.full(self.dim, hi)])

        solver = MetaheuristicClass(func, self.dim, bounds, self.budget)
        x_best = solver.solve()
        x_best = np.clip(np.asarray(x_best, dtype=float), lo, hi)
        return float(func(x_best))

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        # callable_func is the Metaheuristic class (not an instance)
        scores = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                run_bests.append(self._run_one(instance, callable_func, seed))
            scores.append(float(np.log1p(np.mean(run_bests))))
        return float(np.mean(scores))
