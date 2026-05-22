import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData, pareto_front_2d, hypervolume_2d


class Evaluation:
    """Post-hoc evaluator for multi-objective metaheuristics designed by EoH.

    Evaluates on all four ZDT benchmark instances (ZDT1–ZDT4) with multiple
    random seeds, reporting per-instance mean HV, std HV, and Pareto front size.
    """

    def __init__(self, dim: int = 10, budget: int = 5000, n_instances: int = 4,
                 n_runs: int = 5):
        self.dim = dim
        self.budget = budget
        self.n_runs = n_runs
        self.instances = GetData().get_instances(dim=dim, n_instances=n_instances)

    def run_one(self, instance: dict, MetaheuristicClass, seed: int) -> dict:
        """Run one solve trial; return HV and Pareto front statistics."""
        np.random.seed(seed)
        func   = instance['func']
        lo, hi = instance['bounds']          # per-dimension arrays
        bounds = np.array([lo, hi])
        ref_pt = instance['ref_pt']
        n_obj  = instance['n_obj']

        solver = MetaheuristicClass(func, self.dim, bounds, self.budget, n_obj)
        X_front = solver.solve()

        X_front = np.clip(np.asarray(X_front, dtype=float).reshape(-1, self.dim), lo, hi)
        F_front = np.array([func(x) for x in X_front])
        pf      = pareto_front_2d(F_front)
        hv      = hypervolume_2d(F_front, ref_pt)
        return {'hv': hv, 'pareto_size': len(pf)}

    def evaluate(self, MetaheuristicClass) -> list[dict]:
        """Evaluate MetaheuristicClass across all instances and seeds.

        Returns a list of per-instance result dicts with keys:
            name, mean_hv, std_hv, mean_pareto_size
        """
        results = []
        for inst in self.instances:
            hvs, pf_sizes = [], []
            for seed in range(self.n_runs):
                r = self.run_one(inst, MetaheuristicClass, seed)
                hvs.append(r['hv'])
                pf_sizes.append(r['pareto_size'])
            results.append({
                'name':             inst['name'],
                'mean_hv':          float(np.mean(hvs)),
                'std_hv':           float(np.std(hvs)),
                'mean_pareto_size': float(np.mean(pf_sizes)),
            })
        return results
