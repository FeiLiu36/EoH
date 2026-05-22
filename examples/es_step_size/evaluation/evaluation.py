import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData


class Evaluation:
    """Post-hoc evaluator for ES step-size adaptation rules.

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

    def __init__(self, lam: int = 10, max_evals: int = 15000,
                 n_runs: int = 10, ema_alpha: float = 0.2):
        self.lam = lam
        self.max_evals = max_evals
        self.max_generations = (max_evals - 1) // lam
        self.n_runs = n_runs
        self.ema_alpha = ema_alpha

        data = GetData()
        func_map = {
            'sphere':     data.sphere,
            'rastrigin':  data.rastrigin,
            'ackley':     data.ackley,
            'rosenbrock': data.rosenbrock,
            'griewank':   data.griewank,
        }
        self.instances = [{'func': func_map[c['name']], **c} for c in self.CONFIGS]

    def _run_es(self, instance: dict, adapt_fn) -> float:
        func = instance['func']
        n = instance['dim']
        lo, hi = instance['bounds']

        x = lo + (hi - lo) * np.random.rand(n)
        f_x = func(x)
        sigma = (hi - lo) / 4.0
        domain_width = hi - lo

        acceptance_rate = 0.2
        best_f = f_x
        n_evals = 1
        generation = 0

        while n_evals < self.max_evals:
            remaining = self.max_evals - n_evals
            lam_this = min(self.lam, remaining)

            offspring = np.clip(
                x + sigma * np.random.randn(lam_this, n), lo, hi
            )
            f_offspring = np.array([func(o) for o in offspring])
            n_evals += lam_this

            n_accepted = int(np.sum(f_offspring < f_x))
            gen_acceptance = n_accepted / lam_this

            acceptance_rate = ((1 - self.ema_alpha) * acceptance_rate
                               + self.ema_alpha * gen_acceptance)

            best_idx = int(np.argmin(f_offspring))
            if f_offspring[best_idx] < f_x:
                x = offspring[best_idx]
                f_x = f_offspring[best_idx]
                best_f = min(best_f, f_x)

            new_sigma = adapt_fn(
                float(sigma),
                float(acceptance_rate),
                float(f_x),
                f_offspring.copy(),
                n,
                generation,
                self.max_generations,
            )
            sigma = float(np.clip(new_sigma, 1e-12, domain_width))
            generation += 1

        return float(best_f)

    def evaluate(self, adapt_fn) -> list[dict]:
        """Evaluate adapt_fn on the full benchmark suite.

        Returns a list of result dicts, one per (function, dim) combination.
        Each dict has keys: name, dim, mean, std, log1p_mean.
        """
        results = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                best = self._run_es(instance, adapt_fn)
                run_bests.append(best)
            results.append({
                'name':       instance['name'],
                'dim':        instance['dim'],
                'mean':       float(np.mean(run_bests)),
                'std':        float(np.std(run_bests)),
                'log1p_mean': float(np.log1p(np.mean(run_bests))),
            })
        return results
