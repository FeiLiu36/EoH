# Template heuristic: DE/rand/1 (classic baseline).
# Replace the body of `mutation` with the best operator found by EoH.

import numpy as np


def mutation(population: np.ndarray, current_idx: int, best_idx: int,
             fitness: np.ndarray, F: float, bounds: np.ndarray) -> np.ndarray:
    """DE/rand/1 mutation: v = x_r1 + F * (x_r2 - x_r3)."""
    pop_size, dim = population.shape
    candidates = [i for i in range(pop_size) if i != current_idx]
    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
    return population[r1] + F * (population[r2] - population[r3])
