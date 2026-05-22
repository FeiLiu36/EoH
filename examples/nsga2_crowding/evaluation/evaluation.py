import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import _ZDT1, _ZDT2, _ZDT3
from prob import _nds, _sbx, _poly_mut, _get_pareto_front, _hypervolume_2d


class Evaluation:
    """Post-hoc evaluator for NSGA-II crowding-distance operators.

    Uses a larger budget and more diverse ZDT instances than the training
    evaluator in prob.py, and reports per-instance hypervolume alongside
    an overall summary.
    """

    CONFIGS = [
        {'name': 'ZDT1', 'func_cls': _ZDT1, 'n_var': 30, 'n_obj': 2,
         'ref_point': np.array([1.1, 1.1])},
        {'name': 'ZDT2', 'func_cls': _ZDT2, 'n_var': 30, 'n_obj': 2,
         'ref_point': np.array([1.1, 1.1])},
        {'name': 'ZDT3', 'func_cls': _ZDT3, 'n_var': 30, 'n_obj': 2,
         'ref_point': np.array([1.1, 1.1])},
    ]

    def __init__(self, pop_size: int = 100, n_gen: int = 200, n_runs: int = 10):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_runs = n_runs
        self.instances = [
            {**{k: v for k, v in cfg.items() if k != 'func_cls'},
             'func': cfg['func_cls']()}
            for cfg in self.CONFIGS
        ]

    # ------------------------------------------------------------------
    # NSGA-II runner (mirrors prob.py, with extended budget)
    # ------------------------------------------------------------------

    def _run_nsga2(self, instance: dict, crowding_fn, seed: int) -> float:
        rng = np.random.default_rng(seed)
        problem_fn = instance['func']
        n_var = instance['n_var']
        ref_point = instance['ref_point']
        N = self.pop_size

        X = rng.uniform(0.0, 1.0, (N, n_var))
        F = np.array([problem_fn(x) for x in X])

        fronts = _nds(F)
        rank = np.zeros(N, dtype=int)
        crowd = np.zeros(N)
        for r, front in enumerate(fronts):
            rank[front] = r
            n_f = len(front)
            cd = crowding_fn(F[front]) if n_f > 1 else np.full(n_f, np.inf)
            crowd[front] = np.asarray(cd, dtype=float).ravel()

        for _ in range(self.n_gen):
            QX, QF = [], []
            while len(QX) < N:
                a, b = rng.integers(0, N), rng.integers(0, N)
                p1 = a if rank[a] < rank[b] or (rank[a] == rank[b] and crowd[a] >= crowd[b]) else b
                a, b = rng.integers(0, N), rng.integers(0, N)
                p2 = a if rank[a] < rank[b] or (rank[a] == rank[b] and crowd[a] >= crowd[b]) else b
                c1, c2 = _sbx(rng, X[p1], X[p2])
                c1 = _poly_mut(rng, c1)
                c2 = _poly_mut(rng, c2)
                QX.extend([c1, c2])
                QF.extend([problem_fn(c1), problem_fn(c2)])

            QX, QF = np.array(QX[:N]), np.array(QF[:N])
            RX = np.vstack([X, QX])
            RF = np.vstack([F, QF])
            fronts = _nds(RF)

            crowd_r = np.zeros(2 * N)
            for front in fronts:
                n_f = len(front)
                cd = crowding_fn(RF[front]) if n_f > 1 else np.full(n_f, np.inf)
                crowd_r[front] = np.asarray(cd, dtype=float).ravel()

            new_X, new_F = [], []
            for front in fronts:
                if len(new_X) + len(front) <= N:
                    new_X.extend(RX[front].tolist())
                    new_F.extend(RF[front].tolist())
                else:
                    remaining = N - len(new_X)
                    best = sorted(front, key=lambda i: -crowd_r[i])[:remaining]
                    new_X.extend(RX[best].tolist())
                    new_F.extend(RF[best].tolist())
                    break

            X, F = np.array(new_X), np.array(new_F)
            fronts = _nds(F)
            rank = np.zeros(N, dtype=int)
            crowd = np.zeros(N)
            for r, front in enumerate(fronts):
                rank[front] = r
                n_f = len(front)
                cd = crowding_fn(F[front]) if n_f > 1 else np.full(n_f, np.inf)
                crowd[front] = np.asarray(cd, dtype=float).ravel()

        pareto_F = _get_pareto_front(F)
        return _hypervolume_2d(pareto_F, ref_point)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, crowding_fn) -> list[dict]:
        """Evaluate crowding_fn on the full benchmark suite.

        Returns a list of result dicts, one per instance. Each dict has:
            name, n_var, n_obj, hv_mean, hv_std
        """
        results = []
        for instance in self.instances:
            hv_runs = [
                self._run_nsga2(instance, crowding_fn, seed)
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
