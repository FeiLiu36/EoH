import numpy as np


class GetData:
    """ZDT benchmark problem configurations for NSGA-II crossover evaluation.

    Only plain dicts with JSON-serialisable / numpy values are stored here so
    that the task object remains fully picklable when sent to evaluation
    subprocesses.  The actual pymoo Problem objects are created on-demand
    inside _run_nsga2(), which executes inside the already-spawned subprocess.
    """

    def get_instances(self):
        """Return training problem configurations.

        Each entry has:
            name      – pymoo problem identifier passed to get_problem()
            n_var     – number of decision variables (for documentation/mutation rate)
            ref_point – hypervolume reference point (strictly dominates Pareto front)
        """
        return [
            {
                'name': 'zdt1',
                'n_var': 30,
                'ref_point': np.array([1.1, 1.1]),
            },
            {
                'name': 'zdt2',
                'n_var': 30,
                'ref_point': np.array([1.1, 1.1]),
            },
        ]
