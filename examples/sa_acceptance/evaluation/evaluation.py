import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData


class Evaluation:
    """Post-hoc evaluator for SA acceptance probability functions.

    Uses a larger iteration budget, more seeds, and both 10-D and 20-D variants
    of each benchmark compared with the training evaluator in prob.py.
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

    def __init__(self, max_iter: int = 20000, sigma_ratio: float = 0.02,
                 T_ratio: float = 1e-3, n_runs: int = 10):
        self.max_iter = max_iter
        self.sigma_ratio = sigma_ratio
        self.T_ratio = T_ratio
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
        for inst in self.instances:
            inst['T0'] = self._calibrate_T0(inst)

    def _calibrate_T0(self, instance: dict, n_samples: int = 200) -> float:
        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']
        sigma = self.sigma_ratio * (hi - lo)
        np.random.seed(0)
        deltas = []
        for _ in range(n_samples):
            x = lo + (hi - lo) * np.random.rand(dim)
            x_new = np.clip(x + np.random.normal(0, sigma, dim), lo, hi)
            d = func(x_new) - func(x)
            if d > 0:
                deltas.append(d)
        mean_delta = float(np.mean(deltas)) if deltas else 1.0
        return mean_delta / np.log(2)

    def _run_sa(self, instance: dict, acceptance_fn) -> float:
        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']
        sigma = self.sigma_ratio * (hi - lo)
        T0 = instance['T0']
        cooling = self.T_ratio ** (1.0 / self.max_iter)

        x = lo + (hi - lo) * np.random.rand(dim)
        current_f = func(x)
        best_f = current_f
        T = T0

        for it in range(self.max_iter):
            x_new = np.clip(x + np.random.normal(0, sigma, dim), lo, hi)
            new_f = func(x_new)
            delta = new_f - current_f

            if delta < 0:
                x, current_f = x_new, new_f
            else:
                p = float(acceptance_fn(delta, T, it, self.max_iter))
                p = max(0.0, min(1.0, p))
                if np.random.rand() < p:
                    x, current_f = x_new, new_f

            if current_f < best_f:
                best_f = current_f

            T *= cooling

        return float(best_f)

    def evaluate(self, acceptance_fn) -> list[dict]:
        """Evaluate acceptance_fn on the full benchmark suite.

        Returns a list of result dicts with keys: name, dim, mean, std, log1p_mean.
        """
        results = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                run_bests.append(self._run_sa(instance, acceptance_fn))
            results.append({
                'name':       instance['name'],
                'dim':        instance['dim'],
                'mean':       float(np.mean(run_bests)),
                'std':        float(np.std(run_bests)),
                'log1p_mean': float(np.log1p(np.mean(run_bests))),
            })
        return results
