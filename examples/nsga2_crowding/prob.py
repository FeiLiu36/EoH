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


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers (must be at module scope to survive subprocess pickling)
# ──────────────────────────────────────────────────────────────────────────────

def _nds(F: np.ndarray) -> list:
    """Fast non-dominated sorting.  Returns list of fronts (lists of indices)."""
    n = len(F)
    # dominates[i, j] = True if solution i strictly dominates solution j
    dom = (
        np.all(F[:, np.newaxis] <= F[np.newaxis], axis=2)
        & np.any(F[:, np.newaxis] < F[np.newaxis], axis=2)
    )
    n_dom = dom.sum(axis=0).astype(int)   # how many solutions dominate each solution

    fronts: list = []
    assigned = np.zeros(n, dtype=bool)
    while not np.all(assigned):
        mask = ~assigned & (n_dom == 0)
        if not np.any(mask):
            fronts.append(np.where(~assigned)[0].tolist())
            break
        front = np.where(mask)[0]
        fronts.append(front.tolist())
        assigned[front] = True
        n_dom -= dom[np.ix_(front, np.arange(n))].sum(axis=0)
    return fronts


def _sbx(rng, x1: np.ndarray, x2: np.ndarray,
         eta: float = 15.0) -> tuple:
    """Simulated binary crossover for variables in [0, 1]."""
    c1, c2 = x1.copy(), x2.copy()
    for i in range(len(x1)):
        if rng.random() < 0.5 and abs(x1[i] - x2[i]) > 1e-10:
            u = rng.random()
            beta = (2 * u) ** (1.0 / (eta + 1)) if u <= 0.5 \
                else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))
            c1[i] = np.clip(0.5 * ((x1[i] + x2[i]) - beta * abs(x2[i] - x1[i])), 0.0, 1.0)
            c2[i] = np.clip(0.5 * ((x1[i] + x2[i]) + beta * abs(x2[i] - x1[i])), 0.0, 1.0)
    return c1, c2


def _poly_mut(rng, x: np.ndarray, eta: float = 20.0) -> np.ndarray:
    """Polynomial mutation for variables in [0, 1]."""
    y = x.copy()
    for i in range(len(y)):
        if rng.random() < 1.0 / len(y):
            u = rng.random()
            delta = (2 * u) ** (1.0 / (eta + 1)) - 1.0 if u < 0.5 \
                else 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1))
            y[i] = np.clip(y[i] + delta, 0.0, 1.0)
    return y


def _hypervolume_2d(F: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact 2D hypervolume indicator (minimisation)."""
    feasible = F[np.all(F < ref_point, axis=1)]
    if len(feasible) == 0:
        return 0.0
    sf = feasible[np.argsort(feasible[:, 0])]
    # Keep only non-dominated (f2 strictly decreasing as f1 increases)
    nd = [sf[0]]
    for i in range(1, len(sf)):
        if sf[i, 1] < nd[-1][1]:
            nd.append(sf[i])
    nd = np.array(nd)
    hv = 0.0
    for i in range(len(nd)):
        f1_next = nd[i + 1, 0] if i + 1 < len(nd) else ref_point[0]
        hv += (f1_next - nd[i, 0]) * (ref_point[1] - nd[i, 1])
    return float(hv)


def _get_pareto_front(F: np.ndarray) -> np.ndarray:
    """Return non-dominated solutions from F (minimisation)."""
    n = len(F)
    is_dom = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dom[i]:
            continue
        for j in range(n):
            if i != j and not is_dom[j]:
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dom[i] = True
                    break
    return F[~is_dom]


# ──────────────────────────────────────────────────────────────────────────────
# Problem class
# ──────────────────────────────────────────────────────────────────────────────

class NSGA2Crowding(BaseProblem):
    """EoH task: automatically design the crowding-distance operator for NSGA-II.

    The LLM designs `crowding_distance(F) -> np.ndarray`, which computes a
    per-solution diversity score for one non-dominated front.  The harness
    embeds the designed operator into a complete NSGA-II loop and evaluates
    it on ZDT benchmark problems (2 objectives).  Fitness is the negative mean
    hypervolume across instances and seeds (lower is better for EoH, which
    minimises; higher HV means a better Pareto front approximation).
    """

    template_program = '''
import numpy as np
import math

def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Compute a diversity score for each solution in a non-dominated front.

    Args:
        F (np.ndarray): Objective vectors of all solutions in one Pareto front.
                        Shape: (n_solutions, n_objectives)
    Returns:
        np.ndarray: Diversity score per solution.  Higher means the solution
                    is more isolated (more diverse) and should be preferred
                    in the survival / parent selection steps of NSGA-II.
                    Shape: (n_solutions,)
    """
    n, m = F.shape
    dist = np.zeros(n)
    for obj in range(m):
        idx = np.argsort(F[:, obj])
        dist[idx[0]] = np.inf
        dist[idx[-1]] = np.inf
        f_range = F[idx[-1], obj] - F[idx[0], obj]
        if f_range < 1e-10:
            continue
        for k in range(1, n - 1):
            dist[idx[k]] += (F[idx[k + 1], obj] - F[idx[k - 1], obj]) / f_range
    return dist
'''

    task_description = (
        "Design a novel diversity (crowding) metric for the NSGA-II multi-objective "
        "evolutionary algorithm. NSGA-II selects survivors and parents using two "
        "criteria: Pareto dominance rank (primary) and a per-solution diversity score "
        "(secondary tie-breaker). Your function receives the objective vectors F of "
        "all solutions in one non-dominated front and must return a scalar diversity "
        "score for each solution — higher means the solution is more isolated and "
        "should be preferred. The classic approach is the crowding distance (average "
        "side length of the cuboid formed by the two nearest neighbours in each "
        "objective), but you are encouraged to design more adaptive or informative "
        "metrics such as niche-sharing, hypervolume contribution, or nearest-neighbour "
        "distances. Performance is measured by the hypervolume of the final "
        "approximation set on ZDT benchmark problems (2 objectives) — higher is better."
    )

    def __init__(self, pop_size: int = 100, n_gen: int = 100, n_runs: int = 3,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_runs = n_runs
        self.instances = GetData().get_instances()

    # ------------------------------------------------------------------
    # NSGA-II runner
    # ------------------------------------------------------------------

    def _run_nsga2(self, instance: dict, crowding_fn, seed: int) -> float:
        rng = np.random.default_rng(seed)
        problem_fn = instance['func']
        n_var = instance['n_var']
        ref_point = instance['ref_point']
        N = self.pop_size

        # ── Initialise ────────────────────────────────────────────────
        X = rng.uniform(0.0, 1.0, (N, n_var))
        F = np.array([problem_fn(x) for x in X])

        # Precompute initial rank + crowding for parent selection
        fronts = _nds(F)
        rank = np.zeros(N, dtype=int)
        crowd = np.zeros(N)
        for r, front in enumerate(fronts):
            rank[front] = r
            n_f = len(front)
            cd = crowding_fn(F[front]) if n_f > 1 else np.full(n_f, np.inf)
            crowd[front] = np.asarray(cd, dtype=float).ravel()

        for _ in range(self.n_gen):
            # ── Offspring generation ───────────────────────────────────
            QX, QF = [], []
            while len(QX) < N:
                # Binary tournament selection (rank first, then crowding)
                a, b = rng.integers(0, N), rng.integers(0, N)
                p1 = a if rank[a] < rank[b] or (rank[a] == rank[b] and crowd[a] >= crowd[b]) else b
                a, b = rng.integers(0, N), rng.integers(0, N)
                p2 = a if rank[a] < rank[b] or (rank[a] == rank[b] and crowd[a] >= crowd[b]) else b

                c1, c2 = _sbx(rng, X[p1], X[p2])
                c1 = _poly_mut(rng, c1)
                c2 = _poly_mut(rng, c2)
                QX.extend([c1, c2])
                QF.extend([problem_fn(c1), problem_fn(c2)])

            QX = np.array(QX[:N])
            QF = np.array(QF[:N])

            # ── Environmental selection on R = P ∪ Q ──────────────────
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

            X = np.array(new_X)
            F = np.array(new_F)

            # Recompute rank + crowding for next generation
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
    # EoH interface
    # ------------------------------------------------------------------

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        try:
            hv_totals = []
            for instance in self.instances:
                hv_runs = [
                    self._run_nsga2(instance, callable_func, seed)
                    for seed in range(self.n_runs)
                ]
                hv_totals.append(float(np.mean(hv_runs)))
            return float(-np.mean(hv_totals))
        except Exception:
            return None
