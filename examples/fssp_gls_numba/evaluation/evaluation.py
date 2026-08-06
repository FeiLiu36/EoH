import importlib
import sys
import os

import numpy as np

# Re-use the FSSP/GLS engine from prob.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prob import gls, INVALID


class Evaluation:
    """Evaluate an evolved heuristic on a set of Taillard flow-shop instances.

    Reports the relative makespan gap to the best-known upper bound shipped in
    each instance header:

        gap = (makespan / ub - 1) * 100 %

    (lower = better; 0 % means the best-known solution was matched). Instances
    on which the heuristic fails are counted separately and left out of the
    averages, so a partly broken heuristic cannot look good by averaging in
    only its successes.
    """

    def __init__(self, instances, time_max=10.0, iter_max=1000):
        # instances: list of get_instance.Instance tuples
        self.instances = instances
        self.time_max = time_max
        self.iter_max = iter_max

    def evaluate(self, heuristic=None):
        """Run GLS on every instance.

        Args:
            heuristic: the get_matrix_and_jobs function to test. Defaults to
                       the one in heuristic.py next to this file.
        Returns:
            dict with 'gap' and 'makespan' (means over the solved instances),
            the per-instance 'gaps' and 'makespans' (None where GLS failed),
            and 'n_failed'.
        """
        if heuristic is None:
            mod = importlib.reload(importlib.import_module("heuristic"))
            heuristic = mod.get_matrix_and_jobs

        gaps, makespans = [], []
        for inst in self.instances:
            cmax = gls(inst.n_jobs, inst.tasks, inst.n_machines,
                       self.time_max, self.iter_max, heuristic)
            if cmax >= INVALID:
                gaps.append(None)
                makespans.append(None)
            else:
                gaps.append((cmax / inst.ub - 1) * 100)
                makespans.append(cmax)

        solved_gaps = [g for g in gaps if g is not None]
        solved_cmax = [c for c in makespans if c is not None]
        return {
            'gap': float(np.mean(solved_gaps)) if solved_gaps else float('nan'),
            'makespan': float(np.mean(solved_cmax)) if solved_cmax else float('nan'),
            'gaps': gaps,
            'makespans': makespans,
            'n_failed': len(gaps) - len(solved_gaps),
        }
