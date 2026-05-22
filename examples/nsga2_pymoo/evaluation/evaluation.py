import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prob import _CrossoverAdapter


class Evaluation:
    """Post-hoc evaluator for NSGA-II crossover operators (pymoo backend).

    Uses a larger budget and more diverse ZDT instances than the training
    evaluator in prob.py, and reports per-instance hypervolume alongside
    an overall summary.
    """

    CONFIGS = [
        {'name': 'zdt1', 'n_var': 30, 'ref_point': np.array([1.1, 1.1])},
        {'name': 'zdt2', 'n_var': 30, 'ref_point': np.array([1.1, 1.1])},
        {'name': 'zdt3', 'n_var': 30, 'ref_point': np.array([1.1, 1.1])},
    ]

    def __init__(self, pop_size: int = 100, n_gen: int = 200, n_runs: int = 10):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_runs = n_runs

    def _run_nsga2(self, instance: dict, crossover_fn, seed: int) -> float:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.mutation.pm import PM
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        from pymoo.indicators.hv import HV
        from pymoo.problems import get_problem

        problem = get_problem(instance['name'])
        ref_point = instance['ref_point']

        algorithm = NSGA2(
            pop_size=self.pop_size,
            crossover=_CrossoverAdapter(crossover_fn),
            mutation=PM(prob=1.0 / instance['n_var'], eta=20),
            eliminate_duplicates=True,
        )
        termination = get_termination("n_gen", self.n_gen)
        res = minimize(problem, algorithm, termination, seed=seed, verbose=False)
        return float(HV(ref_point=ref_point)(res.opt.get("F")))

    def evaluate(self, crossover_fn) -> list[dict]:
        """Evaluate crossover_fn on the full ZDT benchmark suite.

        Returns a list of result dicts, one per instance. Each dict has:
            name, n_var, hv_mean, hv_std
        """
        results = []
        for cfg in self.CONFIGS:
            hv_runs = [
                self._run_nsga2(cfg, crossover_fn, seed)
                for seed in range(self.n_runs)
            ]
            results.append({
                'name':    cfg['name'].upper(),
                'n_var':   cfg['n_var'],
                'hv_mean': float(np.mean(hv_runs)),
                'hv_std':  float(np.std(hv_runs)),
            })
        return results
