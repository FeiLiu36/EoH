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


class MOEADDecomposition(BaseProblem):
    """EoH task: automatically design the decomposition operator for MOEA/D.

    The LLM designs `custom_decomposition(F, weights, ideal_point) -> np.ndarray`.
    The harness runs a full MOEA/D loop on DTLZ benchmark problems and measures
    the hypervolume (HV) of the final non-dominated set. Fitness returned is the
    negative mean HV across instances and seeds (lower is better for EoH, which
    minimises).
    """

    template_program = '''
import numpy as np
import math
def custom_decomposition(F: np.ndarray,
                         weights: np.ndarray,
                         ideal_point: np.ndarray) -> np.ndarray:
    """Design a novel decomposition method for MOEA/D.

    Args:
        F (np.ndarray): Objective vectors of a batch of solutions.
                        Shape: (n_solutions, n_objectives)
        weights (np.ndarray): Weight vectors for the corresponding subproblems.
                              Shape: (n_solutions, n_objectives)
        ideal_point (np.ndarray): Best objective value found so far per objective.
                                  Shape: (n_objectives,)
    Returns:
        np.ndarray: Scalar aggregation score per solution (lower means the
                    solution is better for the given subproblem).
                    Shape: (n_solutions,)
    """
    # Default: Tchebycheff (Chebyshev) decomposition
    v = np.abs(F - ideal_point) * weights
    return np.max(v, axis=1)
'''

    task_description = (
        "Design a novel decomposition function for the Multi-Objective Evolutionary "
        "Algorithm based on Decomposition (MOEA/D). MOEA/D converts a multi-objective "
        "problem into a set of scalar subproblems using weight vectors; the decomposition "
        "function defines how each subproblem aggregates the objective vector into a "
        "single scalar. Your function receives a batch of objective vectors F "
        "(shape: n_solutions × n_objectives), the corresponding weight vectors "
        "(same shape), and the current ideal point (shape: n_objectives), and must "
        "return a scalar score for each solution (lower = better for that subproblem). "
        "Classic strategies include Tchebycheff (weighted Chebyshev distance from "
        "the ideal point), Weighted Sum, and Penalty-Based Boundary Intersection "
        "(PBI), but you are encouraged to design more adaptive or creative strategies. "
        "Performance is measured by the hypervolume of the final Pareto front on the "
        "DTLZ2 benchmark (3 objectives) — higher hypervolume is better."
    )

    def __init__(self, n_gen: int = 100, n_runs: int = 3, T: int = 5,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_gen = n_gen
        self.n_runs = n_runs
        self.T = T
        self.instances = GetData().get_instances()
        self._weights_cache: dict = {}

    # ------------------------------------------------------------------
    # Weight-vector generation (Das-Dennis normal boundary intersection)
    # ------------------------------------------------------------------

    def _das_dennis_weights(self, n_obj: int, H: int) -> np.ndarray:
        key = (n_obj, H)
        if key in self._weights_cache:
            return self._weights_cache[key]
        weights: list = []

        def _recurse(remaining: int, n_left: int, current: list) -> None:
            if n_left == 1:
                weights.append(current + [remaining / H])
            else:
                for i in range(remaining + 1):
                    _recurse(remaining - i, n_left - 1, current + [i / H])

        _recurse(H, n_obj, [])
        W = np.array(weights, dtype=float)
        W = np.where(W == 0.0, 1e-6, W)  # avoid zero weights
        self._weights_cache[key] = W
        return W

    # ------------------------------------------------------------------
    # Pareto dominance and hypervolume
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pareto_front(F: np.ndarray) -> np.ndarray:
        """Return the non-dominated subset of F (minimisation)."""
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return F[~is_dominated]

    @staticmethod
    def _hypervolume_mc(F: np.ndarray, ref_point: np.ndarray,
                        n_samples: int = 20_000) -> float:
        """Monte Carlo hypervolume estimation (minimisation).

        A uniform random sample in [0, ref_point] is classified as dominated
        if at least one solution in F weakly dominates it in all objectives.
        """
        feasible = F[np.all(F < ref_point, axis=1)]
        if len(feasible) == 0:
            return 0.0
        rng = np.random.default_rng(0)
        samples = rng.uniform(0.0, ref_point, size=(n_samples, len(ref_point)))
        # (n_sol, n_samples): True where feasible[i] dominates samples[k]
        dominated = np.any(
            np.all(feasible[:, np.newaxis, :] <= samples[np.newaxis, :, :], axis=2),
            axis=0,
        )
        return float(np.mean(dominated) * np.prod(ref_point))

    # ------------------------------------------------------------------
    # MOEA/D runner
    # ------------------------------------------------------------------

    def _run_moead(self, instance: dict, decomp_fn, seed: int) -> float:
        """Run one MOEA/D trial and return the hypervolume of the final front."""
        rng = np.random.default_rng(seed)
        problem_fn = instance['func']
        n_var = instance['n_var']
        n_obj = instance['n_obj']
        ref_point = instance['ref_point']

        W = self._das_dennis_weights(n_obj, H=5)   # 21 vectors for n_obj=3
        pop_size = len(W)

        # Precompute neighbourhood (T closest weight vectors by Euclidean distance)
        dists = np.sum((W[:, np.newaxis, :] - W[np.newaxis, :, :]) ** 2, axis=2)
        neighbors = np.argsort(dists, axis=1)[:, : self.T]

        # Initialise population and evaluate
        X = rng.uniform(0.0, 1.0, (pop_size, n_var))
        F_vals = np.array([problem_fn(x) for x in X])   # (pop_size, n_obj)
        z_star = np.min(F_vals, axis=0).copy()           # ideal point

        # Main MOEA/D loop
        for _ in range(self.n_gen):
            perm = rng.permutation(pop_size)
            for i in perm:
                nb = neighbors[i]

                # Reproduction: DE/rand/1 + binomial crossover
                p1_idx, p2_idx = rng.choice(nb, 2, replace=False)
                r_idx = rng.integers(0, pop_size)
                mutant = X[p1_idx] + 0.5 * (X[p2_idx] - X[r_idx])
                cross_mask = rng.random(n_var) < 0.9
                cross_mask[rng.integers(n_var)] = True
                child = np.where(cross_mask, mutant, X[i])
                child = np.clip(child, 0.0, 1.0)

                # Evaluate child
                child_F = problem_fn(child)
                z_star = np.minimum(z_star, child_F)

                # Update neighbours via the LLM-designed decomposition
                F_nb = F_vals[nb]                                    # (T, n_obj)
                W_nb = W[nb]                                         # (T, n_obj)
                child_F_batch = np.tile(child_F, (len(nb), 1))      # (T, n_obj)

                old_scores = np.asarray(
                    decomp_fn(F_nb, W_nb, z_star), dtype=float
                ).ravel()
                new_scores = np.asarray(
                    decomp_fn(child_F_batch, W_nb, z_star), dtype=float
                ).ravel()

                if old_scores.shape != (len(nb),) or new_scores.shape != (len(nb),):
                    raise ValueError(
                        f"decomp_fn returned shape {new_scores.shape}, expected ({len(nb)},)"
                    )

                update_mask = new_scores <= old_scores
                X[nb[update_mask]] = child
                F_vals[nb[update_mask]] = child_F

        pareto_F = self._get_pareto_front(F_vals)
        return self._hypervolume_mc(pareto_F, ref_point)

    # ------------------------------------------------------------------
    # EoH interface
    # ------------------------------------------------------------------

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        try:
            hv_totals = []
            for instance in self.instances:
                hv_runs = [
                    self._run_moead(instance, callable_func, seed)
                    for seed in range(self.n_runs)
                ]
                hv_totals.append(float(np.mean(hv_runs)))
            # EoH minimises; return negative HV so higher HV = lower fitness
            return float(-np.mean(hv_totals))
        except Exception:
            return None
