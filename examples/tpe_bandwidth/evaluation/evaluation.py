import sys
import os
import numpy as np

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData


class Evaluation:
    """Post-hoc evaluator for TPE observation-weighting rules.

    Uses Optuna's actual TPESampler with the designed weights function.
    Larger budget and more seeds than the training evaluator in prob.py;
    also includes wider-domain variants to test generalisation.
    """

    CONFIGS = [
        {'name': 'sphere',    'lo': -5.12,   'hi':  5.12},
        {'name': 'sphere',    'lo': -50.0,   'hi':  50.0},
        {'name': 'rastrigin', 'lo': -5.12,   'hi':  5.12},
        {'name': 'rastrigin', 'lo': -2.0,    'hi':   2.0},
        {'name': 'ackley',    'lo': -32.768, 'hi': 32.768},
        {'name': 'ackley',    'lo': -5.0,    'hi':   5.0},
        {'name': 'griewank',  'lo': -100.0,  'hi': 100.0},
        {'name': 'griewank',  'lo': -10.0,   'hi':  10.0},
        {'name': 'narrow',    'lo':   0.0,   'hi':   1.0},
        {'name': 'narrow',    'lo':  -0.5,   'hi':   1.5},
    ]

    def __init__(self, n_startup: int = 20, n_iter: int = 60,
                 n_ei_candidates: int = 64, n_runs: int = 10):
        self.n_startup = n_startup
        self.n_iter = n_iter
        self.n_ei_candidates = n_ei_candidates
        self.n_runs = n_runs

        data = GetData()
        func_map = {
            'sphere':    data.sphere,
            'rastrigin': data.rastrigin,
            'ackley':    data.ackley,
            'griewank':  data.griewank,
            'narrow':    data.narrow,
        }
        self.instances = [{'func': func_map[c['name']], **c} for c in self.CONFIGS]

    def _run_optuna_tpe(self, instance: dict, weights_fn, seed: int) -> float:
        func = instance['func']
        lo, hi = instance['lo'], instance['hi']

        def objective(trial):
            x = trial.suggest_float('x', lo, hi)
            return float(func(x))

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup,
            n_ei_candidates=self.n_ei_candidates,
            weights=weights_fn,
            seed=seed,
        )
        study = optuna.create_study(sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.n_startup + self.n_iter,
            show_progress_bar=False,
        )
        return float(study.best_value)

    def evaluate(self, weights_fn) -> list[dict]:
        """Evaluate weights_fn on the full benchmark suite.

        Returns a list of result dicts, one per (function, domain) combination.
        Each dict has keys: name, lo, hi, mean, std, log1p_mean.
        """
        results = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                best = self._run_optuna_tpe(instance, weights_fn, seed)
                run_bests.append(best)
            results.append({
                'name':       instance['name'],
                'lo':         instance['lo'],
                'hi':         instance['hi'],
                'mean':       float(np.mean(run_bests)),
                'std':        float(np.std(run_bests)),
                'log1p_mean': float(np.log1p(np.mean(run_bests))),
            })
        return results
