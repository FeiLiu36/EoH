import numpy as np


class _Branin:
    """Branin function scaled to [0, 1]^2.  Global minimum ≈ 0.397887."""

    f_opt: float = 0.397887

    def __call__(self, x: np.ndarray) -> float:
        x1 = -5.0 + 15.0 * x[0]
        x2 = 15.0 * x[1]
        b = 5.1 / (4.0 * np.pi ** 2)
        c = 5.0 / np.pi
        r, s, t = 6.0, 10.0, 1.0 / (8.0 * np.pi)
        return float((x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)


class _Hartmann3:
    """Hartmann-3 function on [0, 1]^3.  Global minimum ≈ -3.86278."""

    f_opt: float = -3.86278

    _A = np.array([[3.0, 10.0, 30.0],
                   [0.1, 10.0, 35.0],
                   [3.0, 10.0, 30.0],
                   [0.1, 10.0, 35.0]])

    _P = 1e-4 * np.array([[3689.0, 1170.0, 2673.0],
                           [4699.0, 4387.0, 7470.0],
                           [1091.0, 8732.0, 5547.0],
                           [381.0,  5743.0, 8828.0]])

    _alpha = np.array([1.0, 1.2, 3.0, 3.2])

    def __call__(self, x: np.ndarray) -> float:
        return float(-np.sum(self._alpha * np.exp(
            -np.sum(self._A * (x - self._P) ** 2, axis=1)
        )))


class _Hartmann6:
    """Hartmann-6 function on [0, 1]^6.  Global minimum ≈ -3.32237."""

    f_opt: float = -3.32237

    _A = np.array([[10.0,  3.0, 17.0,  3.5,  1.7,  8.0],
                   [ 0.05,10.0, 17.0,  0.1,  8.0, 14.0],
                   [ 3.0,  3.5,  1.7, 10.0, 17.0,  8.0],
                   [17.0,  8.0,  0.05,10.0,  0.1, 14.0]])

    _P = 1e-4 * np.array([[1312.0, 1696.0, 5569.0,  124.0, 8283.0, 5886.0],
                           [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
                           [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
                           [4047.0, 8828.0, 8732.0, 5743.0, 1091.0,  381.0]])

    _alpha = np.array([1.0, 1.2, 3.0, 3.2])

    def __call__(self, x: np.ndarray) -> float:
        return float(-np.sum(self._alpha * np.exp(
            -np.sum(self._A * (x - self._P) ** 2, axis=1)
        )))


class GetData:
    """Bayesian optimisation benchmark instances (all minimisation, domain [0,1]^n).

    Named callable classes (not lambdas) are used so the instances remain
    fully picklable when the task object is sent to evaluation subprocesses.
    """

    def get_instances(self):
        """Training instances: Branin (2D) and Hartmann-3 (3D).

        Each entry has:
            name    – human-readable label
            func    – picklable callable f(x) -> float (minimisation target)
            n_var   – number of decision variables
            f_opt   – known global minimum value
        """
        b = _Branin()
        h3 = _Hartmann3()
        return [
            {'name': 'Branin',    'func': b,  'n_var': 2, 'f_opt': b.f_opt},
            {'name': 'Hartmann3', 'func': h3, 'n_var': 3, 'f_opt': h3.f_opt},
        ]
