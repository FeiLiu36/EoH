import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData


class Evaluation:
    """Post-hoc evaluator for DE crossover operators at multiple dimensions.

    Uses a larger budget and more seeds than the training evaluator in prob.py.
    Tests on 50-D, 100-D, and 200-D variants to verify that the evolved
    crossover scales across dimensionalities.
    """

    CONFIGS = [
        {'name': 'sphere',     'dim':  50, 'bounds': (-5.12,   5.12)},
        {'name': 'sphere',     'dim': 100, 'bounds': (-5.12,   5.12)},
        {'name': 'sphere',     'dim': 200, 'bounds': (-5.12,   5.12)},
        {'name': 'rastrigin',  'dim':  50, 'bounds': (-5.12,   5.12)},
        {'name': 'rastrigin',  'dim': 100, 'bounds': (-5.12,   5.12)},
        {'name': 'rastrigin',  'dim': 200, 'bounds': (-5.12,   5.12)},
        {'name': 'ackley',     'dim':  50, 'bounds': (-32.768, 32.768)},
        {'name': 'ackley',     'dim': 100, 'bounds': (-32.768, 32.768)},
        {'name': 'ackley',     'dim': 200, 'bounds': (-32.768, 32.768)},
        {'name': 'rosenbrock', 'dim':  50, 'bounds': (-2.048,  2.048)},
        {'name': 'rosenbrock', 'dim': 100, 'bounds': (-2.048,  2.048)},
        {'name': 'rosenbrock', 'dim': 200, 'bounds': (-2.048,  2.048)},
        {'name': 'griewank',   'dim':  50, 'bounds': (-600.0,  600.0)},
        {'name': 'griewank',   'dim': 100, 'bounds': (-600.0,  600.0)},
        {'name': 'griewank',   'dim': 200, 'bounds': (-600.0,  600.0)},
    ]

    def __init__(self, pop_size: int = 50, max_evals: int = 100000,
                 n_runs: int = 20, F: float = 0.8, CR: float = 0.9):
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

    def _run_de(self, instance: dict, crossover_fn) -> float:
        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']
        bounds = np.column_stack([np.full(dim, lo), np.full(dim, hi)])
        max_gen = self.max_evals // self.pop_size

        pop = lo + (hi - lo) * np.random.rand(self.pop_size, dim)
        fitness = np.array([func(ind) for ind in pop])
        n_evals = self.pop_size
        best_idx = int(np.argmin(fitness))
        generation = 0

        while n_evals < self.max_evals:
            fitness_best = float(fitness[best_idx])

            for i in range(self.pop_size):
                if n_evals >= self.max_evals:
                    break

                candidates = [j for j in range(self.pop_size) if j != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

                trial = crossover_fn(
                    pop[i].copy(), mutant, self.CR,
                    int(generation), int(max_gen),
                    float(fitness[i]), fitness_best,
                )
                trial = np.asarray(trial, dtype=float)
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                trial_fit = func(trial)
                n_evals += 1
                if trial_fit <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < fitness[best_idx]:
                        best_idx = i

            generation += 1

        return float(fitness[best_idx])

    def evaluate(self, crossover_fn) -> list[dict]:
        """Evaluate crossover_fn on the full benchmark suite.

        Returns a list of result dicts, one per (function, dim) combination.
        Each dict has keys: name, dim, mean, std, log1p_mean.
        """
        results = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                best = self._run_de(instance, crossover_fn)
                run_bests.append(best)
            results.append({
                'name':       instance['name'],
                'dim':        instance['dim'],
                'mean':       float(np.mean(run_bests)),
                'std':        float(np.std(run_bests)),
                'log1p_mean': float(np.log1p(np.mean(run_bests))),
            })
        return results
