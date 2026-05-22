# Template heuristic: Rechenberg 1/5 success rule (classic baseline).
# Replace the body of `adapt_step_size` with the best rule found by EoH.

import numpy as np


def adapt_step_size(
    sigma: float,
    acceptance_rate: float,
    f_parent: float,
    f_offspring: np.ndarray,
    n: int,
    generation: int,
    max_generations: int,
) -> float:
    """Rechenberg 1/5 success rule: adjust sigma based on smoothed acceptance rate."""
    c = 0.817  # Rechenberg damping constant
    if acceptance_rate > 0.2:
        return sigma / c   # increase step size
    elif acceptance_rate < 0.2:
        return sigma * c   # decrease step size
    return sigma
