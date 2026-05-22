# Template heuristic: tournament selection (DEAP's default for eaSimple).
# Replace the body of `select` with the best operator found by EoH.

import numpy as np


def select(
    fitnesses: np.ndarray,
    k: int,
    tournament_size: int,
) -> np.ndarray:
    """Tournament selection: run k independent tournaments of size tournament_size."""
    pop_size = len(fitnesses)
    selected = np.empty(k, dtype=int)
    for i in range(k):
        candidates = np.random.choice(pop_size, tournament_size, replace=False)
        selected[i] = candidates[np.argmin(fitnesses[candidates])]
    return selected
