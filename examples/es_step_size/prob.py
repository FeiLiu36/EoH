# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


class ESStepSize(BaseProblem):
    """EoH task: automatically design the step-size adaptation rule for Evolution Strategies.

    The LLM designs `adapt_step_size`. The harness wraps it in a (1+lambda)-ES loop:
    each generation, `lambda` offspring are sampled from N(parent, sigma^2 * I), the
    best offspring that improves on the parent is accepted, and `adapt_step_size` is
    called to update sigma. A smoothed acceptance rate (EMA over recent generations)
    and the raw offspring fitness array are provided so the rule can exploit both
    acceptance-rate signals and fitness-landscape curvature information.

    Fitness is the mean log1p(best_found) across all benchmarks and seeds — lower is better.
    """

    template_program = '''
import numpy as np
import math
def adapt_step_size(
    sigma: float,
    acceptance_rate: float,
    f_parent: float,
    f_offspring: np.ndarray,
    n: int,
    generation: int,
    max_generations: int,
) -> float:
    """Design a step-size adaptation rule for Evolution Strategies.

    Called once per generation after offspring evaluation and parent selection.
    The returned sigma will be used to sample offspring in the next generation.

    Args:
        sigma:           current step size (standard deviation of isotropic Gaussian mutation)
        acceptance_rate: smoothed fraction of offspring that improved on the parent,
                         computed as an exponential moving average (alpha=0.2) over
                         recent generations; starts at 0.2 (the 1/5-success target)
        f_parent:        fitness of the current parent (best solution found so far);
                         lower is better
        f_offspring:     array of shape (lam,) containing raw fitness values of all
                         offspring sampled this generation (before selection); can be
                         used to estimate fitness landscape curvature — e.g.
                         np.std(f_offspring) / sigma**2 approximates the Hessian trace
                         scaled by the step size, since for a quadratic landscape
                         Var[f(x + sigma*z)] ~ sigma^4 * ||H||_F^2
        n:               problem dimensionality
        generation:      current generation index (0-based)
        max_generations: total number of generations planned

    Returns:
        sigma_new: updated step size (positive float); will be clipped to [1e-12, domain_width]
    """
    # Rechenberg 1/5 success rule (classic baseline)
    c = 0.817  # Rechenberg damping constant
    if acceptance_rate > 0.2:
        return sigma / c   # increase step size
    elif acceptance_rate < 0.2:
        return sigma * c   # decrease step size
    return sigma
'''

    task_description = (
        "Design a novel step-size adaptation rule for a (1+lambda)-Evolution Strategy. "
        "Each generation, `lambda` offspring are sampled isotropically from "
        "N(parent, sigma^2 * I) and the best improving offspring is accepted as the new parent. "
        "The adaptation rule receives: the current step size sigma, a smoothed acceptance rate "
        "(fraction of offspring that improved on the parent, tracked as an EMA), the parent "
        "fitness, the full array of offspring fitness values (which encodes fitness landscape "
        "curvature information — e.g., high spread implies steep or curved landscape), the "
        "problem dimensionality, and the current/total generation counts. "
        "The classic Rechenberg 1/5-success rule adjusts sigma by a fixed factor depending "
        "on whether the acceptance rate is above or below 1/5. "
        "You are encouraged to design more adaptive rules that combine acceptance-rate signals "
        "with curvature estimates (derived from the offspring fitness distribution), progress "
        "rate information, or schedule-based annealing. "
        "The goal is to minimise the average final objective value across a suite of "
        "10-dimensional continuous benchmark functions: Sphere, Rastrigin, Ackley, "
        "Rosenbrock, and Griewank."
    )

    def __init__(self, lam: int = 10, max_evals: int = 3000, n_runs: int = 3,
                 ema_alpha: float = 0.2, timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.lam = lam
        self.max_evals = max_evals
        self.max_generations = (max_evals - 1) // lam
        self.n_runs = n_runs
        self.ema_alpha = ema_alpha
        self.instances = GetData().get_instances()

    def _run_es(self, instance: dict, adapt_fn) -> float:
        """Run one (1+lambda)-ES trial with the evolved step-size adaptation rule."""
        func = instance['func']
        n = instance['dim']
        lo, hi = instance['bounds']

        x = lo + (hi - lo) * np.random.rand(n)
        f_x = func(x)
        sigma = (hi - lo) / 4.0
        domain_width = hi - lo

        acceptance_rate = 0.2  # start at the 1/5-success target
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

            # Fraction of offspring that strictly improve on parent
            n_accepted = int(np.sum(f_offspring < f_x))
            gen_acceptance = n_accepted / lam_this

            # EMA-smoothed acceptance rate
            acceptance_rate = ((1 - self.ema_alpha) * acceptance_rate
                               + self.ema_alpha * gen_acceptance)

            # (1+lambda) selection: accept best offspring if it improves
            best_idx = int(np.argmin(f_offspring))
            if f_offspring[best_idx] < f_x:
                x = offspring[best_idx]
                f_x = f_offspring[best_idx]
                best_f = min(best_f, f_x)

            # Evolve step size
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

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        scores = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                best = self._run_es(instance, callable_func)
                run_bests.append(best)
            scores.append(float(np.log1p(np.mean(run_bests))))
        return float(np.mean(scores))
