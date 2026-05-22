import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData


class Evaluation:
    """Post-hoc evaluator for DE mutation operators.

    Uses a larger budget and more seeds than the training evaluator in prob.py,
    and reports per-benchmark results alongside an overall summary.
    """

    # Extended benchmark suite: 10-D and 20-D variants
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

    def __init__(self, pop_size: int = 50, max_evals: int = 20000,
                 n_runs: int = 10, F: float = 0.5, CR: float = 0.9):
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.n_runs = n_runs
        self.F = F
        self.CR = CR

        data = GetData()
        func_map = {
            'sphere':     data.sphere,
            'rastrigin':  data.rastrigin,
            'ackley':     data.ackley,
            'rosenbrock': data.rosenbrock,
            'griewank':   data.griewank,
        }
        self.instances = [{'func': func_map[c['name']], **c} for c in self.CONFIGS]

    def _run_de(self, instance: dict, mutation_fn) -> float:
        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']
        bounds = np.column_stack([np.full(dim, lo), np.full(dim, hi)])

        pop = lo + (hi - lo) * np.random.rand(self.pop_size, dim)
        fitness = np.array([func(ind) for ind in pop])
        n_evals = self.pop_size
        best_idx = int(np.argmin(fitness))

        while n_evals < self.max_evals:
            for i in range(self.pop_size):
                if n_evals >= self.max_evals:
                    break

                mutant = mutation_fn(
                    pop.copy(), i, best_idx, fitness.copy(), self.F, bounds
                )
                mutant = np.asarray(mutant, dtype=float)

                cross_mask = np.random.rand(dim) < self.CR
                cross_mask[np.random.randint(dim)] = True
                trial = np.where(cross_mask, mutant, pop[i])
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                trial_fit = func(trial)
                n_evals += 1
                if trial_fit <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < fitness[best_idx]:
                        best_idx = i

        return float(fitness[best_idx])

    def evaluate(self, mutation_fn) -> list[dict]:
        """Evaluate mutation_fn on the full benchmark suite.

        Returns a list of result dicts, one per (function, dim) combination.
        Each dict has keys: name, dim, mean, std, log1p_mean.
        """
        results = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                best = self._run_de(instance, mutation_fn)
                run_bests.append(best)
            results.append({
                'name':       instance['name'],
                'dim':        instance['dim'],
                'mean':       float(np.mean(run_bests)),
                'std':        float(np.std(run_bests)),
                'log1p_mean': float(np.log1p(np.mean(run_bests))),
            })
        return results
