import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import _DTLZ2


class Evaluation:
    """Post-hoc evaluator for MOEA/D decomposition operators.

    Uses a larger budget and more diverse instances than the training evaluator
    in prob.py, and reports per-instance hypervolume alongside an overall summary.
    """

    # Extended instance suite: DTLZ2 with varying n_var
    CONFIGS = [
        {'name': 'DTLZ2', 'n_var':  7, 'n_obj': 3, 'ref_point': np.array([2.0, 2.0, 2.0])},
        {'name': 'DTLZ2', 'n_var': 12, 'n_obj': 3, 'ref_point': np.array([2.0, 2.0, 2.0])},
        {'name': 'DTLZ2', 'n_var': 20, 'n_obj': 3, 'ref_point': np.array([2.0, 2.0, 2.0])},
        {'name': 'DTLZ2', 'n_var': 30, 'n_obj': 3, 'ref_point': np.array([2.0, 2.0, 2.0])},
    ]

    def __init__(self, n_gen: int = 200, n_runs: int = 10, T: int = 5,
                 hv_samples: int = 30_000):
        self.n_gen = n_gen
        self.n_runs = n_runs
        self.T = T
        self.hv_samples = hv_samples
        self.instances = [
            {**cfg, 'func': _DTLZ2(n_obj=cfg['n_obj'])} for cfg in self.CONFIGS
        ]
        self._weights_cache: dict = {}

    # ------------------------------------------------------------------
    # Weight-vector generation
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
        W = np.where(W == 0.0, 1e-6, W)
        self._weights_cache[key] = W
        return W

    # ------------------------------------------------------------------
    # Pareto dominance and hypervolume
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pareto_front(F: np.ndarray) -> np.ndarray:
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

    def _hypervolume_mc(self, F: np.ndarray, ref_point: np.ndarray) -> float:
        feasible = F[np.all(F < ref_point, axis=1)]
        if len(feasible) == 0:
            return 0.0
        rng = np.random.default_rng(0)
        samples = rng.uniform(0.0, ref_point, size=(self.hv_samples, len(ref_point)))
        dominated = np.any(
            np.all(feasible[:, np.newaxis, :] <= samples[np.newaxis, :, :], axis=2),
            axis=0,
        )
        return float(np.mean(dominated) * np.prod(ref_point))

    # ------------------------------------------------------------------
    # MOEA/D runner
    # ------------------------------------------------------------------

    def _run_moead(self, instance: dict, decomp_fn, seed: int) -> float:
        rng = np.random.default_rng(seed)
        problem_fn = instance['func']
        n_var = instance['n_var']
        n_obj = instance['n_obj']
        ref_point = instance['ref_point']

        W = self._das_dennis_weights(n_obj, H=5)
        pop_size = len(W)

        dists = np.sum((W[:, np.newaxis, :] - W[np.newaxis, :, :]) ** 2, axis=2)
        neighbors = np.argsort(dists, axis=1)[:, :self.T]

        X = rng.uniform(0.0, 1.0, (pop_size, n_var))
        F_vals = np.array([problem_fn(x) for x in X])
        z_star = np.min(F_vals, axis=0).copy()

        for _ in range(self.n_gen):
            perm = rng.permutation(pop_size)
            for i in perm:
                nb = neighbors[i]
                p1_idx, p2_idx = rng.choice(nb, 2, replace=False)
                r_idx = rng.integers(0, pop_size)
                mutant = X[p1_idx] + 0.5 * (X[p2_idx] - X[r_idx])
                cross_mask = rng.random(n_var) < 0.9
                cross_mask[rng.integers(n_var)] = True
                child = np.where(cross_mask, mutant, X[i])
                child = np.clip(child, 0.0, 1.0)

                child_F = problem_fn(child)
                z_star = np.minimum(z_star, child_F)

                nb_F = F_vals[nb]
                nb_W = W[nb]
                child_F_batch = np.tile(child_F, (len(nb), 1))

                old_scores = np.asarray(decomp_fn(nb_F, nb_W, z_star), dtype=float).ravel()
                new_scores = np.asarray(decomp_fn(child_F_batch, nb_W, z_star), dtype=float).ravel()
                update_mask = new_scores <= old_scores
                X[nb[update_mask]] = child
                F_vals[nb[update_mask]] = child_F

        pareto_F = self._get_pareto_front(F_vals)
        return self._hypervolume_mc(pareto_F, ref_point)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, decomp_fn) -> list[dict]:
        """Evaluate decomp_fn on the full benchmark suite.

        Returns a list of result dicts, one per instance. Each dict has:
            name, n_var, n_obj, hv_mean, hv_std
        """
        results = []
        for instance in self.instances:
            hv_runs = [
                self._run_moead(instance, decomp_fn, seed)
                for seed in range(self.n_runs)
            ]
            results.append({
                'name':    instance['name'],
                'n_var':   instance['n_var'],
                'n_obj':   instance['n_obj'],
                'hv_mean': float(np.mean(hv_runs)),
                'hv_std':  float(np.std(hv_runs)),
            })
        return results
