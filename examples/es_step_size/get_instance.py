import numpy as np


class GetData:
    """Classic continuous benchmark functions for evaluating ES step-size adaptation rules.

    All functions have a global optimum of 0.0.
    """

    @staticmethod
    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        n = len(x)
        return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

    @staticmethod
    def ackley(x: np.ndarray) -> float:
        n = len(x)
        return float(
            -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
            + 20 + np.e
        )

    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))

    @staticmethod
    def griewank(x: np.ndarray) -> float:
        n = len(x)
        i = np.arange(1, n + 1, dtype=float)
        return float(np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1)

    def get_instances(self):
        """Return a list of benchmark function configurations.

        Each entry is a dict with keys:
            name   – human-readable label
            func   – callable f(x) → float
            dim    – number of decision variables
            bounds – (lower, upper) scalar pair applied to every dimension
        """
        funcs = {
            'sphere':     self.sphere,
            'rastrigin':  self.rastrigin,
            'ackley':     self.ackley,
            'rosenbrock': self.rosenbrock,
            'griewank':   self.griewank,
        }
        configs = [
            {'name': 'sphere',     'dim': 10, 'bounds': (-5.12,   5.12)},
            {'name': 'rastrigin',  'dim': 10, 'bounds': (-5.12,   5.12)},
            {'name': 'ackley',     'dim': 10, 'bounds': (-32.768, 32.768)},
            {'name': 'rosenbrock', 'dim': 10, 'bounds': (-2.048,  2.048)},
            {'name': 'griewank',   'dim': 10, 'bounds': (-600.0,  600.0)},
        ]
        return [{'func': funcs[c['name']], **c} for c in configs]
