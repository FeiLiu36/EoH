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

from pymoo.core.crossover import Crossover
from eoh import BaseProblem
from get_instance import GetData


# ──────────────────────────────────────────────────────────────────────────────
# pymoo Crossover adapter — module-level so:
#   (a) isinstance(adapter, Crossover) is True (pymoo type check passes)
#   (b) pymoo's internal deepcopy of the algorithm object succeeds
# ──────────────────────────────────────────────────────────────────────────────

class _CrossoverAdapter(Crossover):
    """Wraps a plain (x1, x2) → (c1, c2) function as a pymoo Crossover object."""

    def __init__(self, func):
        super().__init__(n_parents=2, n_offsprings=2)
        self.func = func

    def _do(self, problem, X, **kwargs):
        # X shape: (n_parents=2, n_matings, n_var)
        _, n_matings, _ = X.shape
        Y = np.zeros_like(X)
        for i in range(n_matings):
            c1, c2 = self.func(X[0, i].copy(), X[1, i].copy())
            Y[0, i] = np.clip(np.asarray(c1, dtype=float), problem.xl, problem.xu)
            Y[1, i] = np.clip(np.asarray(c2, dtype=float), problem.xl, problem.xu)
        return Y


class NSGA2Crossover(BaseProblem):
    """EoH task: automatically design the crossover operator for NSGA-II via pymoo.

    The LLM designs `crossover(x1, x2) -> (c1, c2)`, a pairwise recombination
    function for continuous decision vectors.  The harness plugs the designed
    operator into pymoo's NSGA2 algorithm (with standard polynomial mutation,
    binary tournament selection, and rank-and-crowding survival) and evaluates
    the resulting hypervolume on ZDT benchmark problems.  Fitness is the
    negative mean HV across instances and seeds (lower is better for EoH).
    """

    template_program = '''
import numpy as np

def crossover(x1: np.ndarray, x2: np.ndarray) -> tuple:
    """Design a novel crossover operator for NSGA-II continuous optimisation.

    Both inputs are decision vectors with values in [0, 1] (ZDT benchmark
    bounds). Offspring are clipped to [0, 1] automatically after this function
    returns, so there is no need to enforce bounds manually.

    Args:
        x1 (np.ndarray): First parent decision vector.  Shape: (n_var,)
        x2 (np.ndarray): Second parent decision vector.  Shape: (n_var,)
    Returns:
        tuple: (c1, c2) — two offspring arrays of the same shape as the parents.
    """
    # Default: Simulated Binary Crossover (SBX, eta=15)
    eta = 15.0
    c1, c2 = x1.copy(), x2.copy()
    for i in range(len(x1)):
        if np.random.random() < 0.5 and abs(x1[i] - x2[i]) > 1e-10:
            u = np.random.random()
            beta = (2 * u) ** (1.0 / (eta + 1)) if u <= 0.5 \
                else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))
            c1[i] = 0.5 * ((x1[i] + x2[i]) - beta * abs(x2[i] - x1[i]))
            c2[i] = 0.5 * ((x1[i] + x2[i]) + beta * abs(x2[i] - x1[i]))
    return c1, c2
'''

    task_description = (
        "Design a novel crossover operator for the NSGA-II multi-objective "
        "evolutionary algorithm, implemented using the pymoo framework. "
        "The crossover function receives two parent decision vectors "
        "(each of length n_var, with values in [0, 1]) and must return two "
        "offspring vectors of the same length. It is embedded in pymoo's NSGA-II "
        "(with standard polynomial mutation, binary tournament selection, and "
        "rank-and-crowding survival selection) and tested on ZDT benchmark "
        "problems (2 objectives, 30 decision variables). "
        "The classic operator is SBX (Simulated Binary Crossover), but you are "
        "encouraged to design more adaptive or creative strategies such as "
        "differential-evolution-style recombination, blend crossover (BLX-α), "
        "or fitness-aware operators. "
        "Performance is measured by the hypervolume of the final non-dominated "
        "approximation set — higher hypervolume is better."
    )

    def __init__(self, pop_size: int = 100, n_gen: int = 100, n_runs: int = 3,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_runs = n_runs
        self.instances = GetData().get_instances()

    # ------------------------------------------------------------------
    # NSGA-II runner (pymoo)
    # ------------------------------------------------------------------

    def _run_nsga2(self, instance: dict, crossover_fn, seed: int) -> float:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.mutation.pm import PM
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        from pymoo.indicators.hv import HV
        from pymoo.problems import get_problem

        problem = get_problem(instance['name'])
        ref_point = instance['ref_point']

        algorithm = NSGA2(
            pop_size=self.pop_size,
            crossover=_CrossoverAdapter(crossover_fn),
            mutation=PM(prob=1.0 / instance['n_var'], eta=20),
            eliminate_duplicates=True,
        )
        termination = get_termination("n_gen", self.n_gen)
        res = minimize(problem, algorithm, termination, seed=seed, verbose=False)

        hv_calc = HV(ref_point=ref_point)
        return float(hv_calc(res.opt.get("F")))

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
