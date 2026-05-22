import numpy as np


class _ZDT1:
    """ZDT1: convex Pareto front.  f2 = 1 - sqrt(f1),  f1 in [0, 1]."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])


class _ZDT2:
    """ZDT2: concave Pareto front.  f2 = 1 - f1^2,  f1 in [0, 1]."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2])


class _ZDT3:
    """ZDT3: discontinuous Pareto front (five separate segments)."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1))
        return np.array([f1, f2])


class GetData:
    """ZDT benchmark problem instances for evaluating NSGA-II crowding operators.

    All variables in [0, 1]^n_var, 2 objectives, well-studied Pareto fronts.
    Named callable classes (not lambdas) are used so the instances can be
    pickled when the problem object is sent to evaluation subprocesses.
    """

    def get_instances(self):
        """Return training problem configurations.

        Each entry has:
            name      – identifier string
            func      – picklable callable f(x: np.ndarray) -> np.ndarray
            n_var     – number of decision variables
            n_obj     – number of objectives
            ref_point – HV reference point
        """
        return [
            {
                'name': 'ZDT1',
                'func': _ZDT1(),
                'n_var': 30,
                'n_obj': 2,
                'ref_point': np.array([1.1, 1.1]),
            },
            {
                'name': 'ZDT2',
                'func': _ZDT2(),
                'n_var': 30,
                'n_obj': 2,
                'ref_point': np.array([1.1, 1.1]),
            },
        ]
