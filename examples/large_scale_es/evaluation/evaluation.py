import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData


class Evaluation:
    """Post-hoc evaluator for sep-CMA-ES diagonal variance adaptation rules.

    Uses a larger evaluation budget, more seeds, and both n=100 and n=200
    variants of each benchmark to test high-dimensional generalization.
    """

    CONFIGS = [
        {'name': 'sphere',     'dim': 100, 'bounds': (-5.12,   5.12)},
        {'name': 'sphere',     'dim': 200, 'bounds': (-5.12,   5.12)},
        {'name': 'rastrigin',  'dim': 100, 'bounds': (-5.12,   5.12)},
        {'name': 'rastrigin',  'dim': 200, 'bounds': (-5.12,   5.12)},
        {'name': 'ackley',     'dim': 100, 'bounds': (-32.768, 32.768)},
        {'name': 'ackley',     'dim': 200, 'bounds': (-32.768, 32.768)},
        {'name': 'rosenbrock', 'dim': 100, 'bounds': (-2.048,  2.048)},
        {'name': 'rosenbrock', 'dim': 200, 'bounds': (-2.048,  2.048)},
        {'name': 'griewank',   'dim': 100, 'bounds': (-600.0,  600.0)},
        {'name': 'griewank',   'dim': 200, 'bounds': (-600.0,  600.0)},
    ]

    def __init__(self, max_evals: int = 60_000, n_runs: int = 10):
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

    def _sepcmaes_params(self, n: int) -> dict:
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
        weights = raw_w / raw_w.sum()
        mueff = float(1.0 / np.sum(weights ** 2))
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
        return dict(lam=lam, mu=mu, weights=weights, mueff=mueff,
                    cs=cs, damps=damps, cc=cc, c1=c1, cmu=cmu, chi_n=chi_n)

    def _run_sepcmaes(self, instance: dict, adapt_fn) -> float:
        func = instance['func']
        n = instance['dim']
        lo, hi = instance['bounds']
        domain = hi - lo
        p = self._sepcmaes_params(n)
        lam, mu = p['lam'], p['mu']
        weights, mueff = p['weights'], p['mueff']
        cs, damps, cc = p['cs'], p['damps'], p['cc']
        c1, cmu, chi_n = p['c1'], p['cmu'], p['chi_n']
        max_gen = (self.max_evals - 1) // lam

        m = lo + domain * np.random.rand(n)
        sigma = domain / 4.0
        d = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        best_f = np.inf
        n_evals = 0
        generation = 0

        while n_evals < self.max_evals:
            z = np.random.randn(lam, n)
            y = z * np.sqrt(d)
            x = np.clip(m + sigma * y, lo, hi)

            f_vals = np.array([func(xi) for xi in x])
            n_evals += lam

            idx = np.argsort(f_vals)
            best_f = min(best_f, float(f_vals[idx[0]]))
            y_sel = y[idx[:mu]]
            y_w = weights @ y_sel

            m = np.clip(m + sigma * y_w, lo, hi)

            inv_sqrt_d = 1.0 / np.sqrt(np.maximum(d, 1e-20))
            p_sigma = ((1.0 - cs) * p_sigma
                       + np.sqrt(cs * (2.0 - cs) * mueff) * inv_sqrt_d * y_w)
            norm_ps = float(np.linalg.norm(p_sigma))

            adj = norm_ps / np.sqrt(max(1e-20, 1.0 - (1.0 - cs) ** (2.0 * (generation + 1))))
            hsig = 1.0 if adj < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0

            p_c = ((1.0 - cc) * p_c
                   + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w)

            d_new = adapt_fn(
                d.copy(), p_c.copy(), weights.copy(), y_sel.copy(),
                float(c1), float(cmu), float(cc), float(hsig),
                int(n), int(generation), int(max_gen),
            )
            d = np.clip(np.asarray(d_new, dtype=float), 1e-20, 1e10)

            sigma *= float(np.exp((cs / damps) * (norm_ps / chi_n - 1.0)))
            sigma = float(np.clip(sigma, 1e-12, domain))
            generation += 1

        return float(best_f)

    def evaluate(self, adapt_fn) -> list[dict]:
        """Evaluate adapt_fn on the full benchmark suite.

        Returns a list of result dicts with keys: name, dim, mean, std, log1p_mean.
        """
        results = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                run_bests.append(self._run_sepcmaes(instance, adapt_fn))
            results.append({
                'name':       instance['name'],
                'dim':        instance['dim'],
                'mean':       float(np.mean(run_bests)),
                'std':        float(np.std(run_bests)),
                'log1p_mean': float(np.log1p(np.mean(run_bests))),
            })
        return results
