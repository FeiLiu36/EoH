# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import sys
import os
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


class BOAcquisition(BaseProblem):
    """EoH task: automatically design the acquisition function for Bayesian Optimisation.

    The LLM designs `acquisition(mu, sigma, f_best) -> np.ndarray`, which
    scores a batch of candidate points given the GP posterior.  The harness
    runs a Gaussian-Process-based BO loop (sklearn Matérn-5/2 surrogate) on
    classic benchmark functions and measures the log10 simple regret
    (f_best_found − f_opt).  Lower regret → better acquisition function →
    lower fitness (EoH minimises).
    """

    template_program = '''
import numpy as np

def acquisition(mu: np.ndarray, sigma: np.ndarray, f_best: float) -> np.ndarray:
    """Design a novel acquisition function for Bayesian Optimisation.

    The BO loop evaluates the true function at argmax(acquisition(...)).
    All benchmark functions are minimisation problems (lower f is better).

    Args:
        mu (np.ndarray): GP predictive mean for each candidate point.
                         Shape: (n_candidates,).  Lower values are more promising.
        sigma (np.ndarray): GP predictive standard deviation.
                            Shape: (n_candidates,).  Larger = more uncertain.
        f_best (float): Lowest (best) objective value observed so far.
    Returns:
        np.ndarray: Acquisition score per candidate (higher → evaluate next).
                    Shape: (n_candidates,)

    Hint: scipy is available. Use scipy.stats for normal CDF/PDF:
        from scipy.stats import norm
        cdf(z) = norm.cdf(z)   pdf(z) = norm.pdf(z)
    Pure-numpy equivalents (scipy.special.erf):
        from scipy.special import erf
        cdf(z) = 0.5 * (1 + erf(z / np.sqrt(2)))
        pdf(z) = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    """
    # Default: Lower Confidence Bound — explore uncertain, low-mean regions
    kappa = 2.0
    return -mu + kappa * sigma
'''

    task_description = (
        "Design a novel acquisition function for Bayesian Optimisation (BO). "
        "The acquisition function guides the BO loop by scoring a set of "
        "candidate points given the Gaussian Process posterior: it receives "
        "the GP predictive mean `mu` and standard deviation `sigma` for each "
        "candidate, and the current best observed value `f_best` (minimisation "
        "setting — lower objective is better). It must return a scalar score "
        "per candidate; the candidate with the highest score is evaluated next. "
        "Classic strategies include Lower Confidence Bound (LCB: -mu + κ·σ), "
        "Expected Improvement (EI), and Probability of Improvement (PI), but "
        "you are encouraged to design more adaptive or novel strategies — for "
        "example, adaptive κ schedules, portfolio combinations, or information-"
        "theoretic criteria. "
        "Performance is measured by the log10 simple regret (best found value "
        "minus the global optimum) on Branin (2D) and Hartmann-3 (3D) "
        "benchmarks — lower regret is better."
    )

    def __init__(self, n_init: int = 5, n_iter: int = 20, n_candidates: int = 256,
                 n_runs: int = 5, timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_init = n_init
        self.n_iter = n_iter
        self.n_candidates = n_candidates
        self.n_runs = n_runs
        self.instances = GetData().get_instances()

    # ------------------------------------------------------------------
    # Bayesian Optimisation loop
    # ------------------------------------------------------------------

    def _run_bo(self, instance: dict, acq_fn, seed: int) -> float:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel

        rng = np.random.default_rng(seed)
        func = instance['func']
        n_var = instance['n_var']
        f_opt = instance['f_opt']

        # Initial random design
        X_obs = rng.uniform(0.0, 1.0, (self.n_init, n_var))
        y_obs = np.array([func(x) for x in X_obs])

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=1.0, length_scale_bounds=(1e-2, 10.0), nu=2.5
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, n_restarts_optimizer=0, normalize_y=True
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(self.n_iter):
                gp.fit(X_obs, y_obs)

                # Evaluate acquisition on random candidates
                X_cand = rng.uniform(0.0, 1.0, (self.n_candidates, n_var))
                mu, sigma = gp.predict(X_cand, return_std=True)
                sigma = np.maximum(sigma, 1e-9)

                acq_vals = np.asarray(
                    acq_fn(mu, sigma, float(np.min(y_obs))), dtype=float
                ).ravel()

                if acq_vals.shape != (self.n_candidates,):
                    raise ValueError(
                        f"acquisition must return shape ({self.n_candidates},), "
                        f"got {acq_vals.shape}"
                    )

                x_next = X_cand[np.argmax(acq_vals)]
                y_obs = np.append(y_obs, func(x_next))
                X_obs = np.vstack([X_obs, x_next])

        simple_regret = float(np.min(y_obs)) - f_opt
        return max(simple_regret, 0.0)

    # ------------------------------------------------------------------
    # EoH interface
    # ------------------------------------------------------------------

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        try:
            log_regrets = []
            for instance in self.instances:
                regrets = [
                    self._run_bo(instance, callable_func, seed)
                    for seed in range(self.n_runs)
                ]
                # Log10 of mean regret per instance (scale-invariant across functions)
                log_regrets.append(float(np.log10(np.mean(regrets) + 1e-8)))
            return float(np.mean(log_regrets))
        except Exception:
            return None
