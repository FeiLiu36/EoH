"""Post-hoc evaluator for (1+1)-ES mutation noise generators.

Runs nevergrad's _EolOnePlusOne (evolved step) and nevergrad's stock
OnePlusOne (baseline) on a larger 10-D / 20-D benchmark suite.
"""

import sys
import os
import numpy as np
import nevergrad as ng

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData
from prob import _EolOnePlusOne


class Evaluation:
    """Evaluate a generate_mutation function via nevergrad's optimizer engine."""

    CONFIGS = [
        {'name': 'sphere',     'dim': 10, 'bounds': (-5.12,   5.12)},
        {'name': 'sphere',     'dim': 20, 'bounds': (-5.12,   5.12)},
        {'name': 'rastrigin',  'dim': 10, 'bounds': (-5.12,   5.12)},
        {'name': 'rastrigin',  'dim': 20, 'bounds': (-5.12,   5.12)},
        {'name': 'ackley',     'dim': 10, 'bounds': (-32.768, 32.768)},
        {'name': 'ackley',     'dim': 20, 'bounds': (-32.768, 32.768)},
        {'name': 'rosenbrock', 'dim': 10, 'bounds': (-2.048,  2.048)},
        {'name': 'rosenbrock', 'dim': 20, 'bounds': (-2.048,  2.048)},
        {'name': 'griewank',   'dim': 10, 'bounds': (-600.0,  600.0)},
        {'name': 'griewank',   'dim': 20, 'bounds': (-600.0,  600.0)},
    ]

    def __init__(self, max_evals: int = 5000, n_runs: int = 10):
        self.max_evals = max_evals
        self.n_runs = n_runs

        data = GetData()
        func_map = {
            'sphere':     data.sphere,
            'rastrigin':  data.rastrigin,
            'ackley':     data.ackley,
            'rosenbrock': data.rosenbrock,
            'griewank':   data.griewank,
        }
        self.instances = [{'func': func_map[c['name']], **c} for c in self.CONFIGS]

    def _run_evolved(self, instance: dict, mutation_fn) -> float:
        """Run _EolOnePlusOne (nevergrad subclass) with the evolved mutation."""
        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']

        param = ng.p.Array(shape=(dim,)).set_bounds(lo, hi)
        opt = _EolOnePlusOne(parametrization=param, budget=self.max_evals,
                             mutation='gaussian')
        opt._eol_mutation_fn = mutation_fn
        rec = opt.minimize(func)
        return float(func(rec.value))

    def _run_nevergrad(self, instance: dict) -> float:
        """Run nevergrad's stock OnePlusOne as reference."""
        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']

        param = ng.p.Array(shape=(dim,)).set_bounds(lo, hi)
        opt = ng.optimizers.OnePlusOne(parametrization=param, budget=self.max_evals)
        rec = opt.minimize(func)
        return float(func(rec.value))

    def evaluate(self, mutation_fn) -> list[dict]:
        """Evaluate mutation_fn via nevergrad on the full benchmark suite.

        Returns a list of result dicts with keys:
            name, dim, mean, std, log1p_mean.
        """
        results = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                best = self._run_evolved(instance, mutation_fn)
                run_bests.append(best)
            results.append({
                'name':       instance['name'],
                'dim':        instance['dim'],
                'mean':       float(np.mean(run_bests)),
                'std':        float(np.std(run_bests)),
                'log1p_mean': float(np.log1p(np.mean(run_bests))),
            })
        return results

    def evaluate_nevergrad_baseline(self) -> list[dict]:
        """Evaluate nevergrad's stock OnePlusOne on the full benchmark suite."""
        results = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                best = self._run_nevergrad(instance)
                run_bests.append(best)
            results.append({
                'name':       instance['name'],
                'dim':        instance['dim'],
                'mean':       float(np.mean(run_bests)),
                'std':        float(np.std(run_bests)),
                'log1p_mean': float(np.log1p(np.mean(run_bests))),
            })
        return results
