# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import sys
import os
import numpy as np

import nevergrad as ng
from nevergrad.optimization.optimizerlib import _OnePlusOne

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


class _EolOnePlusOne(_OnePlusOne):
    """Nevergrad's _OnePlusOne with a pluggable mutation noise generator.

    Drop-in replacement for nevergrad's standard Gaussian OnePlusOne:
    only the step-generation line is overridden; sigma adaptation
    (×2.0 on improvement, ×0.84 otherwise) and all other machinery
    remain exactly as in nevergrad's source.

    Set `._eol_mutation_fn` on an instance before calling `minimize()`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eol_mutation_fn = None
        self._eol_success_window = 20
        self._eol_history: list[int] = []

    def _internal_ask_candidate(self):
        # Only intercept the Gaussian mutation path; fall back for all others.
        if (self._eol_mutation_fn is None
                or not self._num_ask
                or self.mutation != 'gaussian'):
            return super()._internal_ask_candidate()

        ref = self.parametrization
        pessimistic = self.current_bests['pessimistic'].parameter.spawn_child()
        # For ng.p.Array with set_bounds (default clipping), standardized == value.
        current_x = pessimistic.get_standardized_data(reference=ref)

        success_rate = (
            float(np.mean(self._eol_history)) if self._eol_history else 0.2
        )

        noise = self._eol_mutation_fn(
            current_x.copy(),
            float(self._sigma),
            success_rate,
            self.dimension,
            self._num_ask,
            self.budget or 1000,
        )
        noise = np.asarray(noise, dtype=float)
        if noise.shape != (self.dimension,):
            raise ValueError(
                f"generate_mutation shape {noise.shape} != ({self.dimension},)"
            )

        # Mirrors nevergrad's Gaussian branch:
        #   out = pessimistic.set_standardized_data(self._sigma * step)
        # Here `noise` already encodes the full displacement (sigma * direction).
        out = pessimistic.set_standardized_data(noise)
        out._meta['sigma'] = self._sigma
        return out

    def _internal_tell(self, x, loss):
        # Record success BEFORE the parent updates _previous_best_loss and sigma.
        improved = loss < self._previous_best_loss
        self._eol_history.append(1 if improved else 0)
        if len(self._eol_history) > self._eol_success_window:
            self._eol_history.pop(0)
        super()._internal_tell(x, loss)


class OnePlusOne(BaseProblem):
    """(1+1)-ES — mutation noise design via nevergrad's OnePlusOne engine.

    The LLM designs generate_mutation, the key step in nevergrad's
    _OnePlusOne._internal_ask_candidate (the Gaussian mutation branch).
    The evolved function is plugged into _EolOnePlusOne, a thin subclass
    that overrides only the step-generation line; sigma adaptation
    (nevergrad's exact ×2.0 / ×0.84 rule) and all other mechanics are
    inherited unchanged from nevergrad.

    nevergrad's Gaussian OnePlusOne default:
        step = sigma * N(0, I_d)
    Evaluated with nevergrad's minimize() interface; fitness =
    mean log1p(best_found) across all benchmarks and seeds (lower = better).
    """

    template_program = '''
def generate_mutation(current_solution: np.ndarray, sigma: float,
                      success_rate: float, n_dims: int,
                      iteration: int, max_evals: int) -> np.ndarray:
    """Generate the perturbation vector for nevergrad's (1+1)-ES.

    This function replaces the step-generation line in nevergrad's
    _OnePlusOne._internal_ask_candidate (Gaussian branch):
        step = rng.normal(0, 1, dimension)
        candidate = pessimistic + sigma * step   (in standardized space)

    Args:
        current_solution: (n_dims,) current best solution (standardized ==
                          value space for default nevergrad bounds clipping)
        sigma:            nevergrad's current isotropic step size;
                          adapted ×2.0 on improvement, ×0.84 otherwise
        success_rate:     fraction of successful mutations in the last 20 steps
        n_dims:           problem dimensionality
        iteration:        nevergrad's num_ask counter (0-based)
        max_evals:        total evaluation budget
    Returns:
        noise: (n_dims,) full displacement vector (= sigma * direction);
               added directly to current_solution by nevergrad internals
    """
    # nevergrad OnePlusOne default: isotropic Gaussian displacement
    return sigma * np.random.normal(0.0, 1.0, n_dims)
'''

    task_description = (
        "Design the mutation noise generator for nevergrad's OnePlusOne "
        "optimiser ((1+1)-ES). "
        "The function replaces the Gaussian step in nevergrad's "
        "_OnePlusOne._internal_ask_candidate; a candidate is formed as "
        "y = current_solution + noise and accepted if f(y) < f(current_solution). "
        "Nevergrad adapts sigma via its exact ×2.0 / ×0.84 rule. "
        "The function receives the current best solution, nevergrad's current "
        "sigma, the recent success rate, dimensionality, and iteration progress. "
        "Nevergrad's default is isotropic Gaussian noise sigma * N(0, I), but "
        "creative alternatives — Cauchy tails for rugged landscapes, "
        "coordinate-wise scaling, mixed distributions, or success-rate-adaptive "
        "strategies — may achieve faster convergence or better final quality. "
        "The goal is to minimise the average final objective value across a suite "
        "of 10-dimensional continuous benchmarks: Sphere, Rastrigin, Ackley, "
        "Rosenbrock, and Griewank."
    )

    def __init__(self, max_evals: int = 1000, n_runs: int = 3,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.max_evals = max_evals
        self.n_runs = n_runs
        self.instances = GetData().get_instances()

    def _run_opo(self, instance: dict, mutation_fn) -> float:
        """Run one nevergrad _EolOnePlusOne trial; return best objective found."""
        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']

        param = ng.p.Array(shape=(dim,)).set_bounds(lo, hi)
        opt = _EolOnePlusOne(
            parametrization=param,
            budget=self.max_evals,
            mutation='gaussian',
        )
        opt._eol_mutation_fn = mutation_fn
        recommendation = opt.minimize(func)
        return float(func(recommendation.value))

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        scores = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                try:
                    best = self._run_opo(instance, callable_func)
                except Exception:
                    return None
                run_bests.append(best)
            scores.append(float(np.log1p(np.mean(run_bests))))
        return float(np.mean(scores))
