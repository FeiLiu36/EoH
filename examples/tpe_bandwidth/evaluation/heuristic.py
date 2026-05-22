# Template heuristic: Optuna's built-in default_weights (baseline).
# Replace the body of `compute_weights` with the best rule found by EoH.

import numpy as np


def compute_weights(n: int) -> np.ndarray:
    """Optuna's default_weights: top-25 get weight 1, rest get a linear ramp."""
    if n == 0:
        return np.array([])
    elif n < 25:
        return np.ones(n)
    else:
        ramp = np.linspace(1.0 / n, 1.0, num=n - 25)
        flat = np.ones(25)
        return np.concatenate([ramp, flat])
