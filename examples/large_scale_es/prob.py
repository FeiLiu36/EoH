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


class LargeScaleES(BaseProblem):
    """EoH task: design the diagonal variance adaptation for a high-dimensional ES.

    The algorithm is a separable (μ/μ_W, λ)-CMA-ES (sep-CMA-ES) operating on
    n=100-dimensional problems. Unlike full CMA-ES, which stores an n×n covariance
    matrix, sep-CMA-ES maintains only n scalar variance factors d[i] — one per
    dimension — making it tractable at n=100 to n=10,000.

    Search distribution: x = m + σ · (√d ⊙ z),  z ~ N(0, I_n)

    The LLM designs `adapt_diagonal_cov`, called once per generation to update
    the per-dimension variance vector d. The harness handles mean update, cumulative
    step-size adaptation (CSA), and the evolution path p_c. This is the exact
    analogue of the CMA-ES covariance update but restricted to the diagonal.

    Fitness: mean log1p(best_found) across all 100-D benchmarks and seeds
             (lower is better).
    """

    template_program = '''
import numpy as np
def adapt_diagonal_cov(
    d: np.ndarray,
    p_c: np.ndarray,
    weights: np.ndarray,
    y_k: np.ndarray,
    c1: float,
    cmu: float,
    cc: float,
    hsig: float,
    n: int,
    generation: int,
    max_generations: int,
) -> np.ndarray:
    """Update the per-dimension variance vector for sep-CMA-ES.

    Called once per generation after offspring evaluation and selection.

    Args:
        d:               float array (n,) — current per-dimension variance factors
                         (positive); search std in dimension i is σ · √d[i]
        p_c:             float array (n,) — cumulative evolution path, accumulated
                         from the weighted mean steps y_w of past generations;
                         tracks the historically preferred search direction per dim
        weights:         float array (mu,) — positive recombination weights summing
                         to 1 (log-linear, largest weight for best individual)
        y_k:             float array (mu, n) — normalized steps of the top-mu
                         selected offspring: y_k[i,j] = (x_{i:λ,j} − m_j) / σ;
                         y_k[i,j] ~ N(0, d[j]) under the current distribution
        c1:              rank-1 learning rate (≈ 2/n² for large n)
        cmu:             rank-mu learning rate (≈ mueff/n² for large n)
        cc:              evolution-path accumulation rate (≈ 4/n for large n)
        hsig:            stagnation indicator — 1.0 if evolution path is reliable,
                         0.0 if stagnation is detected (path too long)
        n:               problem dimensionality (100 during training)
        generation:      current generation index (0-based)
        max_generations: total planned generations

    Returns:
        d_new: float array (n,) — updated variance factors (must be strictly positive;
               the harness clips values below 1e-20 and above 1e10)
    """
    # Standard separable rank-1 + rank-mu update (sep-CMA-ES baseline)
    rank1  = c1  * (p_c ** 2 + (1 - hsig) * cc * (2 - cc) * d)
    rankmu = cmu * np.einsum('i,ij->j', weights, y_k ** 2)
    return (1 - c1 - cmu) * d + rank1 + rankmu
'''

    task_description = (
        "Design a novel per-dimension variance adaptation rule for a separable "
        "(μ/μ_W, λ)-CMA-ES (sep-CMA-ES) applied to high-dimensional (n=100) continuous "
        "optimisation. In sep-CMA-ES the full n×n covariance matrix is replaced by "
        "n scalar variance factors d[i], making the algorithm practical for n≥100. "
        "The function updates d given: the current variances d, the cumulative evolution "
        "path p_c (tracking historically preferred directions per dimension), "
        "recombination weights, the normalized steps of the top-μ selected offspring "
        "y_k ~ N(0, d[j]) per dimension, the standard CMA-ES learning rates c1/cmu/cc, "
        "a stagnation indicator hsig, and the generation progress. "
        "The standard baseline applies a rank-1 update (using p_c²) and a rank-mu "
        "update (using the weighted mean of y_k²). "
        "Alternatives include: progress-aware learning rate schedules (large rates "
        "early for fast exploration, small rates late for fine-tuning), regularisation "
        "to prevent variance collapse, log-space updates for numerical stability, "
        "momentum terms that integrate curvature from multiple past generations, "
        "or dimension-wise restarts triggered by stagnation signals. "
        "The goal is to minimise the average final objective across 100-dimensional "
        "benchmark functions: Sphere, Rastrigin, Ackley, Rosenbrock, and Griewank."
    )

    def __init__(self, dim: int = 100, max_evals: int = 30_000, n_runs: int = 3,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.dim = dim
        self.max_evals = max_evals
        self.n_runs = n_runs
        self.instances = GetData().get_instances(dim=dim)

    def _sepcmaes_params(self, n: int) -> dict:
        """Compute standard sep-CMA-ES hyperparameters for dimensionality n."""
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
        """Run one sep-CMA-ES trial and return the best objective value found."""
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
            # Sample offspring: x = m + σ · (√d ⊙ z)
            z = np.random.randn(lam, n)
            y = z * np.sqrt(d)               # y ~ N(0, diag(d)), shape (lam, n)
            x = np.clip(m + sigma * y, lo, hi)

            f_vals = np.array([func(xi) for xi in x])
            n_evals += lam

            idx = np.argsort(f_vals)
            best_f = min(best_f, float(f_vals[idx[0]]))
            y_sel = y[idx[:mu]]              # (mu, n) top-μ steps
            y_w = weights @ y_sel            # (n,) weighted mean step

            # Mean update
            m = np.clip(m + sigma * y_w, lo, hi)

            # Cumulative step-size adaptation (CSA)
            # For diagonal C = diag(d): C^{-1/2} y = y / √d
            inv_sqrt_d = 1.0 / np.sqrt(np.maximum(d, 1e-20))
            p_sigma = ((1.0 - cs) * p_sigma
                       + np.sqrt(cs * (2.0 - cs) * mueff) * inv_sqrt_d * y_w)
            norm_ps = float(np.linalg.norm(p_sigma))

            # Stagnation indicator (using unbiased norm estimate)
            adj = norm_ps / np.sqrt(max(1e-20, 1.0 - (1.0 - cs) ** (2.0 * (generation + 1))))
            hsig = 1.0 if adj < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0

            # Evolution path for diagonal adaptation
            p_c = ((1.0 - cc) * p_c
                   + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w)

            # LLM-designed diagonal variance update
            d_new = adapt_fn(
                d.copy(), p_c.copy(), weights.copy(), y_sel.copy(),
                float(c1), float(cmu), float(cc), float(hsig),
                int(n), int(generation), int(max_gen),
            )
            d_new = np.asarray(d_new, dtype=float)
            if d_new.shape != (n,):
                raise ValueError(
                    f"adapt_diagonal_cov returned shape {d_new.shape}, expected ({n},)"
                )
            d = np.clip(d_new, 1e-20, 1e10)

            # Step-size update via CSA
            sigma *= float(np.exp((cs / damps) * (norm_ps / chi_n - 1.0)))
            sigma = float(np.clip(sigma, 1e-12, domain))
            generation += 1

        return float(best_f)

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        scores = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                run_bests.append(self._run_sepcmaes(instance, callable_func))
            scores.append(float(np.log1p(np.mean(run_bests))))
        return float(np.mean(scores))
