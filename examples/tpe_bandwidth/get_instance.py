import numpy as np


class GetData:
    """1-D benchmark functions for evaluating TPE bandwidth rules.

    Each function has a global minimum of 0.0, so simple regret equals
    the best value found (log1p-transformed for scale invariance).
    """

    @staticmethod
    def sphere(x: float) -> float:
        return float(x ** 2)

    @staticmethod
    def rastrigin(x: float) -> float:
        return float(10.0 + x ** 2 - 10.0 * np.cos(2.0 * np.pi * x))

    @staticmethod
    def ackley(x: float) -> float:
        return float(
            -20.0 * np.exp(-0.2 * abs(x))
            - np.exp(np.cos(2.0 * np.pi * x))
            + 20.0 + np.e
        )

    @staticmethod
    def griewank(x: float) -> float:
        return float(x ** 2 / 4000.0 - np.cos(x) + 1.0)

    @staticmethod
    def narrow(x: float) -> float:
        """Sharp optimum at x=0.3; tests exploitation of a narrow good region."""
        return float(1.0 - np.exp(-200.0 * (x - 0.3) ** 2))

    def get_instances(self):
        """Return a list of 1-D benchmark configurations.

        Each entry is a dict with keys:
            name  – human-readable label
            func  – callable f(x: float) → float
            lo    – lower bound of the search domain
            hi    – upper bound of the search domain
        """
        return [
            {'name': 'sphere',   'func': self.sphere,   'lo': -5.12,   'hi':  5.12},
            {'name': 'rastrigin','func': self.rastrigin, 'lo': -5.12,   'hi':  5.12},
            {'name': 'ackley',   'func': self.ackley,   'lo': -32.768, 'hi': 32.768},
            {'name': 'griewank', 'func': self.griewank,  'lo': -100.0,  'hi': 100.0},
            {'name': 'narrow',   'func': self.narrow,    'lo':  0.0,    'hi':  1.0},
        ]
