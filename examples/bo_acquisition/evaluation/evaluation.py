import sys
import os
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import _Branin, _Hartmann3, _Hartmann6


class Evaluation:
    """Post-hoc evaluator for BO acquisition functions.

    Uses a larger budget (more iterations, more seeds) and an additional
    Hartmann-6 (6D) instance compared to the training evaluator in prob.py.
    Reports per-instance log10 simple regret and an overall summary.
    """

    CONFIGS = [
        {'name': 'Branin',    'cls': _Branin,    'n_var': 2},
        {'name': 'Hartmann3', 'cls': _Hartmann3, 'n_var': 3},
        {'name': 'Hartmann6', 'cls': _Hartmann6, 'n_var': 6},
    ]

    def __init__(self, n_init: int = 10, n_iter: int = 40,
                 n_candidates: int = 512, n_runs: int = 20):
        self.n_init = n_init
        self.n_iter = n_iter
        self.n_candidates = n_candidates
        self.n_runs = n_runs
        self.instances = [
            {'name': c['name'], 'func': c['cls'](), 'f_opt': c['cls'].f_opt,
             'n_var': c['n_var']}
            for c in self.CONFIGS
        ]

    def _run_bo(self, instance: dict, acq_fn, seed: int) -> float:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel

        rng = np.random.default_rng(seed)
        func = instance['func']
        n_var = instance['n_var']
        f_opt = instance['f_opt']

        X_obs = rng.uniform(0.0, 1.0, (self.n_init, n_var))
        y_obs = np.array([func(x) for x in X_obs])

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=1.0, length_scale_bounds=(1e-2, 10.0), nu=2.5
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, n_restarts_optimizer=2, normalize_y=True
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(self.n_iter):
                gp.fit(X_obs, y_obs)
                X_cand = rng.uniform(0.0, 1.0, (self.n_candidates, n_var))
                mu, sigma = gp.predict(X_cand, return_std=True)
                sigma = np.maximum(sigma, 1e-9)
                acq_vals = np.asarray(
                    acq_fn(mu, sigma, float(np.min(y_obs))), dtype=float
                ).ravel()
                x_next = X_cand[np.argmax(acq_vals)]
                y_obs = np.append(y_obs, func(x_next))
                X_obs = np.vstack([X_obs, x_next])

        return max(float(np.min(y_obs)) - f_opt, 0.0)

    def evaluate(self, acq_fn) -> list[dict]:
        """Evaluate acq_fn on the full benchmark suite.

        Returns a list of result dicts, one per instance. Each dict has:
            name, n_var, log_regret_mean, log_regret_std, regret_mean, regret_std
        """
        results = []
        for instance in self.instances:
            regrets = [
                self._run_bo(instance, acq_fn, seed)
                for seed in range(self.n_runs)
            ]
            log_regrets = [float(np.log10(r + 1e-8)) for r in regrets]
            results.append({
                'name':             instance['name'],
                'n_var':            instance['n_var'],
                'regret_mean':      float(np.mean(regrets)),
                'regret_std':       float(np.std(regrets)),
                'log_regret_mean':  float(np.mean(log_regrets)),
                'log_regret_std':   float(np.std(log_regrets)),
            })
        return results
