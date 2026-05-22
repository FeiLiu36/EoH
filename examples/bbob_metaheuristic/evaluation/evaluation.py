import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData


class Evaluation:
    """Post-hoc evaluator for complete black-box metaheuristics.

    Uses a larger evaluation budget and more seeds than the training evaluator
    in prob.py, and tests both 10-D and 20-D variants of each benchmark to
    assess generalization across dimensionalities.
    """

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

    def __init__(self, budget: int = 5000, n_runs: int = 10):
        self.budget = budget
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

    def _run_one(self, instance: dict, MetaheuristicClass, seed: int) -> float:
        np.random.seed(seed)
        func = instance['func']
        lo, hi = instance['bounds']
        dim = instance['dim']
        bounds = np.array([np.full(dim, lo), np.full(dim, hi)])

        solver = MetaheuristicClass(func, dim, bounds, self.budget)
        x_best = solver.solve()
        x_best = np.clip(np.asarray(x_best, dtype=float), lo, hi)
        return float(func(x_best))

    def evaluate(self, MetaheuristicClass) -> list[dict]:
        """Evaluate MetaheuristicClass on the full benchmark suite.

        The class must implement: __init__(self, func, dim, bounds, budget) and solve(self) -> x_best

        Returns a list of result dicts with keys:
            name, dim, mean, std, log1p_mean.
        """
        results = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                run_bests.append(self._run_one(instance, MetaheuristicClass, seed))
            results.append({
                'name':       instance['name'],
                'dim':        instance['dim'],
                'mean':       float(np.mean(run_bests)),
                'std':        float(np.std(run_bests)),
                'log1p_mean': float(np.log1p(np.mean(run_bests))),
            })
        return results
