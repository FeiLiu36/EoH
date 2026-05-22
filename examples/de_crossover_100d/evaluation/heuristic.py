# Template heuristic: standard binomial crossover (DE/bin baseline).
# Replace the body of `crossover` with the best operator found by EoH.

import numpy as np


def crossover(target: np.ndarray, mutant: np.ndarray, CR: float,
              generation: int, max_generations: int,
              fitness_target: float, fitness_best: float) -> np.ndarray:
    """Binomial crossover: each dimension from mutant with probability CR."""
    dim = len(target)
    mask = np.random.rand(dim) < CR
    mask[np.random.randint(dim)] = True
    return np.where(mask, mutant, target)
