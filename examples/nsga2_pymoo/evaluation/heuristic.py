# Template: SBX crossover (standard NSGA-II baseline).
# Replace the body of `crossover` with the best operator found by EoH.

import numpy as np


def crossover(x1: np.ndarray, x2: np.ndarray) -> tuple:
    """Simulated Binary Crossover (SBX, eta=15) — standard NSGA-II baseline.

    Replace this with the best crossover operator discovered by EoH.
    """
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
