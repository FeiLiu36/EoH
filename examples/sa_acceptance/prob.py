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


class SAAcceptance(BaseProblem):
    """EoH task: automatically design the acceptance probability function for SA.

    SA is applied to continuous benchmark minimisation (same five functions as the
    DE mutation example). The harness provides a fixed Gaussian perturbation
    neighbourhood and a geometric temperature schedule. T0 is calibrated per
    benchmark so the classic Boltzmann criterion starts with ~50 % acceptance of
    uphill moves, making the function inputs meaningful across all benchmarks.

    The LLM designs only `acceptance_probability`. Fitness is the mean
    log1p(best_found) across all benchmarks and seeds — lower is better.
    """

    template_program = '''
def acceptance_probability(delta_fitness: float, temperature: float,
                           iteration: int, max_iterations: int) -> float:
    """Decide whether to accept a worse solution in Simulated Annealing.

    Called only when the candidate is worse than the current solution
    (delta_fitness > 0). Return a probability in [0, 1]; the harness
    accepts the candidate if np.random.rand() < returned value.

    Args:
        delta_fitness:  f(candidate) - f(current) > 0 — how much worse the
                        candidate is. Calibrated so the Boltzmann criterion
                        exp(-delta_fitness / temperature) ≈ 0.5 at the start.
        temperature:    current annealing temperature; starts at T0 (problem-
                        specific, >0) and decreases geometrically to T0 * 1e-3.
        iteration:      current iteration index (0-based)
        max_iterations: total number of SA iterations planned
    Returns:
        probability in [0, 1] of accepting the worse candidate
    """
    return float(np.exp(-delta_fitness / max(temperature, 1e-10)))
'''

    task_description = (
        "Design a novel acceptance probability function for the Simulated Annealing (SA) "
        "algorithm applied to continuous optimisation. "
        "The function is called whenever a candidate solution is worse than the current "
        "one (delta_fitness > 0) and must return a probability in [0, 1] of accepting it. "
        "The classic Boltzmann criterion exp(-delta_fitness / temperature) is the baseline. "
        "Alternatives include modified exponents, sigmoid-based criteria, "
        "threshold acceptance, non-monotone acceptance, or adaptive rules that "
        "change behaviour based on iteration progress. "
        "The temperature starts high (encouraging exploration) and decreases "
        "geometrically by a factor of 1000 over all iterations (encouraging exploitation). "
        "The goal is to minimise the average final objective value across a suite of "
        "10-dimensional benchmark functions: Sphere, Rastrigin, Ackley, Rosenbrock, "
        "and Griewank."
    )

    def __init__(self, max_iter: int = 5000, sigma_ratio: float = 0.02,
                 T_ratio: float = 1e-3, n_runs: int = 3,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.max_iter = max_iter
        self.sigma_ratio = sigma_ratio  # step size = sigma_ratio * (hi - lo)
        self.T_ratio = T_ratio          # T_final = T0 * T_ratio
        self.n_runs = n_runs
        self.instances = GetData().get_instances()
        for inst in self.instances:
            inst['T0'] = self._calibrate_T0(inst)

    def _calibrate_T0(self, instance: dict, n_samples: int = 200) -> float:
        """Return T0 so that exp(-mean_uphill_delta / T0) == 0.5.

        This ensures the Boltzmann criterion starts with ~50% acceptance of
        uphill moves, giving a consistent temperature scale across benchmarks.
        """
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
        return mean_delta / np.log(2)  # exp(-mean_delta / T0) = 0.5

    def _run_sa(self, instance: dict, acceptance_fn) -> float:
        """Run one SA trial and return the best objective value found."""
        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']
        sigma = self.sigma_ratio * (hi - lo)
        T0 = instance['T0']
        cooling = (T0 * self.T_ratio / T0) ** (1.0 / self.max_iter)  # == T_ratio^(1/max_iter)

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

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        scores = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                run_bests.append(self._run_sa(instance, callable_func))
            scores.append(float(np.log1p(np.mean(run_bests))))
        return float(np.mean(scores))
