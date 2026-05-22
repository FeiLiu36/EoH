import numpy as np


class _DTLZ2:
    """Picklable callable for the DTLZ2 benchmark (sphere Pareto front).

    Stored as a named class so multiprocessing can pickle it when the problem
    object is sent to evaluation subprocesses (lambdas are not picklable).
    """

    def __init__(self, n_obj: int = 3):
        self.n_obj = n_obj

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n_obj = self.n_obj
        g = float(np.sum((x[n_obj - 1:] - 0.5) ** 2))
        F = np.empty(n_obj)
        F[0] = (1.0 + g) * np.prod(np.cos(x[:n_obj - 1] * np.pi / 2.0))
        for m in range(1, n_obj - 1):
            F[m] = (
                (1.0 + g)
                * np.prod(np.cos(x[:n_obj - 1 - m] * np.pi / 2.0))
                * np.sin(x[n_obj - 1 - m] * np.pi / 2.0)
            )
        F[n_obj - 1] = (1.0 + g) * np.sin(x[0] * np.pi / 2.0)
        return F


class GetData:
    """DTLZ benchmark problem instances for evaluating MOEA/D decomposition operators.

    Provides instances of DTLZ2 (unimodal, sphere-shaped Pareto front) with
    3 objectives, suitable for differentiating decomposition strategies within
    a short run budget.
    """

    def get_instances(self):
        """Return problem configurations for MOEA/D decomposition evaluation.

        Each entry has:
            name      – identifier string
            func      – picklable callable f(x: np.ndarray) -> np.ndarray
            n_var     – number of decision variables
            n_obj     – number of objectives
            ref_point – HV reference point (strictly dominates all Pareto-optimal solutions)
        """
        return [
            {
                'name': 'DTLZ2_3obj_7var',
                'func': _DTLZ2(n_obj=3),
                'n_var': 7,
                'n_obj': 3,
                'ref_point': np.array([2.0, 2.0, 2.0]),
            },
            {
                'name': 'DTLZ2_3obj_12var',
                'func': _DTLZ2(n_obj=3),
                'n_var': 12,
                'n_obj': 3,
                'ref_point': np.array([2.0, 2.0, 2.0]),
            },
        ]
