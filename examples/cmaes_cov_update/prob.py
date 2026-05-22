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


class CMAESCovUpdate(BaseProblem):
    """EoH task: automatically design the covariance matrix update rule for CMA-ES.

    The LLM designs `update_covariance`. The harness wraps it in a standard
    CMA-ES loop (sampling, selection, mean update, CSA step-size adaptation)
    and replaces only the covariance update step with the evolved function.
    Fitness is the mean log1p(best_found) across all benchmarks and seeds —
    lower is better.
    """

    template_program = '''
import numpy as np
import math
def update_covariance(
    C: np.ndarray,
    p_c: np.ndarray,
    weights: np.ndarray,
    y_k: np.ndarray,
    c1: float,
    cmu: float,
    cc: float,
    hsig: float,
    n: int,
) -> np.ndarray:
    """Update the covariance matrix for CMA-ES (standard rank-1 + rank-mu).

    Args:
        C:       (n, n) current covariance matrix.
        p_c:     (n,)   evolution path for covariance adaptation.
        weights: (mu,)  positive recombination weights summing to 1.
        y_k:     (mu, n) top-mu normalised offspring steps: (x_{i:lam} - m_old) / sigma_old.
        c1:      learning rate for the rank-1 update.
        cmu:     learning rate for the rank-mu update.
        cc:      learning rate used to accumulate p_c (needed for the correction term).
        hsig:    1 if evolution path is reliable, 0 if stagnation is detected.
        n:       problem dimensionality.

    Returns:
        C_new: (n, n) updated covariance matrix.
    """
    # Rank-1 update: reinforce the direction of the evolution path
    rank1 = c1 * (np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * C)
    # Rank-mu update: reinforce directions of the selected steps
    rankmu = cmu * np.sum(
        [weights[i] * np.outer(y_k[i], y_k[i]) for i in range(len(weights))], axis=0
    )
    return (1 - c1 - cmu) * C + rank1 + rankmu
'''

    task_description = (
        "Design a novel covariance matrix update rule for the CMA-ES "
        "(Covariance Matrix Adaptation Evolution Strategy) optimisation algorithm. "
        "The update function receives the current covariance matrix C, the evolution "
        "path p_c, the recombination weights, the normalised selected offspring steps "
        "y_k, the rank-1 learning rate c1, the rank-mu learning rate cmu, the "
        "evolution-path learning rate cc, the stagnation flag hsig, and the problem "
        "dimensionality n. It must return an updated (n, n) covariance matrix. "
        "The standard CMA-ES uses a rank-1 plus rank-mu update, but you are encouraged "
        "to design more adaptive or creative rules that exploit the population geometry, "
        "eigenstructure, or iteration information to improve convergence. "
        "The goal is to minimise the average final objective value across a suite of "
        "10-dimensional continuous benchmark functions: Sphere, Rastrigin, Ackley, "
        "Rosenbrock, and Griewank."
    )

    def __init__(self, max_evals: int = 2000, n_runs: int = 3,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.max_evals = max_evals
        self.n_runs = n_runs
        self.instances = GetData().get_instances()

    def _run_cmaes(self, instance: dict, update_cov_fn) -> float:
        """Run one CMA-ES trial with the evolved covariance update rule."""
        func = instance['func']
        n = instance['dim']
        lo, hi = instance['bounds']

        # Standard CMA-ES parameters (Hansen 2016)
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
            # Eigendecompose C for sampling and inversion
            eigvals, B = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)           # shape (n,)
            invsqrt_C = (B / D) @ B.T     # C^{-1/2}

            # Sample lam offspring
            z = np.random.randn(lam, n)
            y = z * D @ B.T               # shape (lam, n); each row ~ N(0, C)
            x = m + sigma * y
            x = np.clip(x, lo, hi)

            f_vals = np.array([func(xi) for xi in x])
            n_evals += lam
            generation += 1

            idx = np.argsort(f_vals)
            best_f = min(best_f, float(f_vals[idx[0]]))

            y_sel = y[idx[:mu]]           # (mu, n) normalised selected steps

            # Update mean
            m_old = m.copy()
            m = weights @ y_sel * sigma + m_old
            y_w = weights @ y_sel         # (n,) weighted mean step

            # Cumulative step-size adaptation (p_sigma)
            p_sigma = ((1 - cs) * p_sigma
                       + np.sqrt(cs * (2 - cs) * mueff) * (invsqrt_C @ y_w))
            norm_ps = float(np.linalg.norm(p_sigma))

            # h_sig: stagnation indicator
            thresh = (1.4 + 2 / (n + 1)) * chi_n
            norm_ps_adj = norm_ps / np.sqrt(1 - (1 - cs) ** (2 * generation))
            hsig = 1.0 if norm_ps_adj < thresh else 0.0

            # Evolution path for covariance
            p_c = ((1 - cc) * p_c
                   + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w)

            # Evolved covariance update
            C_new = update_cov_fn(
                C.copy(), p_c.copy(), weights.copy(), y_sel.copy(),
                float(c1), float(cmu), float(cc), hsig, n,
            )
            C_new = np.asarray(C_new, dtype=float)
            if C_new.shape != (n, n):
                raise ValueError(f"update_covariance returned shape {C_new.shape}, expected ({n}, {n})")
            C = (C_new + C_new.T) / 2  # enforce symmetry

            # Step-size update
            sigma *= float(np.exp((cs / damps) * (norm_ps / chi_n - 1)))
            sigma = float(np.clip(sigma, 1e-12, 1e6))

        return float(best_f)

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        scores = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                best = self._run_cmaes(instance, callable_func)
                run_bests.append(best)
            scores.append(float(np.log1p(np.mean(run_bests))))
        return float(np.mean(scores))
