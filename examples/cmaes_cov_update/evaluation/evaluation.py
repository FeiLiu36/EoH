import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData


class Evaluation:
    """Post-hoc evaluator for CMA-ES covariance update rules.

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

    def __init__(self, max_evals: int = 10000, n_runs: int = 10):
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

    def _run_cmaes(self, instance: dict, update_cov_fn) -> float:
        func = instance['func']
        n = instance['dim']
        lo, hi = instance['bounds']

        lam = 4 + int(3 * np.log(n))
        mu = lam // 2

        weights_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
        weights = weights_raw / weights_raw.sum()
        mueff = weights.sum() ** 2 / (weights ** 2).sum()

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        m = lo + (hi - lo) * np.random.rand(n)
        sigma = (hi - lo) / 3.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        best_f = np.inf
        n_evals = 0
        generation = 0

        while n_evals < self.max_evals:
            eigvals, B = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            invsqrt_C = (B / D) @ B.T

            z = np.random.randn(lam, n)
            y = z * D @ B.T
            x = m + sigma * y
            x = np.clip(x, lo, hi)

            f_vals = np.array([func(xi) for xi in x])
            n_evals += lam
            generation += 1

            idx = np.argsort(f_vals)
            best_f = min(best_f, float(f_vals[idx[0]]))

            y_sel = y[idx[:mu]]
            m_old = m.copy()
            m = weights @ y_sel * sigma + m_old
            y_w = weights @ y_sel

            p_sigma = ((1 - cs) * p_sigma
                       + np.sqrt(cs * (2 - cs) * mueff) * (invsqrt_C @ y_w))
            norm_ps = float(np.linalg.norm(p_sigma))

            thresh = (1.4 + 2 / (n + 1)) * chi_n
            norm_ps_adj = norm_ps / np.sqrt(1 - (1 - cs) ** (2 * generation))
            hsig = 1.0 if norm_ps_adj < thresh else 0.0

            p_c = ((1 - cc) * p_c
                   + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w)

            C_new = update_cov_fn(
                C.copy(), p_c.copy(), weights.copy(), y_sel.copy(),
                float(c1), float(cmu), float(cc), hsig, n,
            )
            C_new = np.asarray(C_new, dtype=float)
            C = (C_new + C_new.T) / 2

            sigma *= float(np.exp((cs / damps) * (norm_ps / chi_n - 1)))
            sigma = float(np.clip(sigma, 1e-12, 1e6))

        return float(best_f)

    def evaluate(self, update_cov_fn) -> list[dict]:
        """Evaluate update_cov_fn on the full benchmark suite.

        Returns a list of result dicts, one per (function, dim) combination.
        Each dict has keys: name, dim, mean, std, log1p_mean.
        """
        results = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                best = self._run_cmaes(instance, update_cov_fn)
                run_bests.append(best)
            results.append({
                'name':       instance['name'],
                'dim':        instance['dim'],
                'mean':       float(np.mean(run_bests)),
                'std':        float(np.std(run_bests)),
                'log1p_mean': float(np.log1p(np.mean(run_bests))),
            })
        return results
