# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import sys
import os
import numpy as np

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


class TPESamplerWeights(BaseProblem):
    """EoH task: automatically design the observation-weighting function for Optuna's TPE.

    Optuna's default sampler is the Tree-structured Parzen Estimator (TPE).
    Each iteration TPE splits past trials into a 'good' group (gamma(n) lowest
    objective values) and a 'bad' group, then fits a Gaussian mixture model (GMM)
    to each group. The `weights` callable — passed directly to
    `optuna.samplers.TPESampler(weights=...)` — controls how much each observation
    in the 'good' group contributes to its GMM: higher weight → more influence on
    the density estimate.

    Observations are ordered worst-to-best within the good group (ascending by
    objective value), so weights[0] corresponds to the worst and weights[-1] to
    the best in that group.

    Optuna's built-in default (`default_weights`) gives equal weight (1.0) to the
    top 25 observations and a linear ramp from 1/n down to 1/n * n/25 for the rest.

    The harness creates a real `optuna.Study` per trial, using a `TPESampler`
    whose `weights` argument is the designed function. Fitness is the mean
    log1p(best_value) across all benchmarks and seeds — lower is better.
    """

    template_program = '''
import numpy as np

def compute_weights(n: int) -> np.ndarray:
    """Design the observation-weighting scheme for Optuna\'s TPE \'good\' group.

    This function is passed directly as `weights=compute_weights` to
    `optuna.samplers.TPESampler`. It is called once per TPE iteration with
    the size of the \'good\' (below-threshold) group and must return a
    non-negative weight array that Optuna uses to build the Parzen estimator.

    Args:
        n: Number of observations in the \'good\' group (gamma(n_total) trials
           with the lowest objective values). May be 0 if no trials exist yet.
           Observations are ordered ascending by objective value, so index 0 is
           the worst and index n-1 is the best in the \'good\' group.

    Returns:
        weights: non-negative array of shape (n,). Larger values give that
                 observation more influence on the Parzen estimator density.
                 Optuna normalises the weights internally, so only relative
                 magnitudes matter. Return an empty array when n == 0.
    """
    # Optuna\'s built-in default_weights (baseline)
    if n == 0:
        return np.array([])
    elif n < 25:
        return np.ones(n)
    else:
        ramp = np.linspace(1.0 / n, 1.0, num=n - 25)
        flat = np.ones(25)
        return np.concatenate([ramp, flat])
'''

    task_description = (
        "Design the observation-weighting function for the 'good' group Parzen estimator "
        "in Optuna's Tree-structured Parzen Estimator (TPE) sampler. "
        "Each TPE iteration splits completed trials into a 'good' group (the gamma(n) "
        "trials with the lowest objective values, where gamma(n) = min(ceil(0.1*n), 25)) "
        "and a 'bad' group. It fits a Gaussian mixture model (GMM) to each group, and "
        "picks the next hyperparameter value by maximising log l(x) - log g(x). "
        "The designed function `compute_weights(n)` is passed directly as the `weights` "
        "argument to `optuna.samplers.TPESampler(weights=compute_weights)`. It receives "
        "the size n of the 'good' group and must return a non-negative weight array of "
        "shape (n,). Observations are ordered ascending by objective value (worst first, "
        "best last), so assigning higher weight to later indices concentrates the density "
        "around the best past points (exploitation), while uniform weights give equal "
        "influence to all good observations (more exploration). "
        "Optuna's built-in default gives weight 1 to the top 25 and a linear ramp for "
        "the rest. You are encouraged to design more adaptive rules — e.g. exponential "
        "decay toward worse observations, rank-based inverse weighting, softmax-like "
        "schemes, or schedules that tighten as n grows. "
        "The goal is to minimise the average best objective value found by Optuna across "
        "five 1-D benchmark functions: Sphere, Rastrigin, Ackley, Griewank, and a "
        "sharp-optimum Narrow function."
    )

    def __init__(self, n_startup: int = 10, n_iter: int = 30,
                 n_ei_candidates: int = 64, n_runs: int = 3,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_startup = n_startup
        self.n_iter = n_iter
        self.n_ei_candidates = n_ei_candidates
        self.n_runs = n_runs
        self.instances = GetData().get_instances()

    def _run_optuna_tpe(self, instance: dict, weights_fn, seed: int) -> float:
        """Run one Optuna TPE study with the designed weights function."""
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

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        try:
            scores = []
            for instance in self.instances:
                run_bests = []
                for seed in range(self.n_runs):
                    best = self._run_optuna_tpe(instance, callable_func, seed)
                    run_bests.append(best)
                scores.append(float(np.log1p(np.mean(run_bests))))
            return float(np.mean(scores))
        except Exception:
            return None
