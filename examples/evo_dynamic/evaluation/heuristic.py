# example heuristic
# replace it with your own heuristic designed by EoH
import numpy as np


def respond_to_change(population: np.ndarray, fitness: np.ndarray,
                      best_position: np.ndarray,
                      bounds: np.ndarray) -> np.ndarray:
    """Random immigrants + elite preservation.

    Keep the top 20 % of individuals (lightly perturbed) and replace the
    remaining 80 % with uniformly-distributed random immigrants.
    This balances memory (elite cluster near old optimum) with diversity
    (immigrants spread across the search space).
    """
    pop_size, n_dims = population.shape
    lower, upper = bounds[0], bounds[1]
    new_pop = np.empty_like(population)

    n_elite = max(1, pop_size // 5)
    elite_idx = np.argsort(fitness)[-n_elite:]  # highest fitness = closest to old optimum

    # Lightly perturb elite members so they spread around the old best
    sigma = (upper - lower).mean() * 0.05
    for k in range(n_elite):
        new_pop[k] = np.clip(
            population[elite_idx[k]] + np.random.normal(0.0, sigma, n_dims),
            lower, upper,
        )

    # Fill remainder with uniformly random immigrants
    new_pop[n_elite:] = np.random.uniform(lower, upper, (pop_size - n_elite, n_dims))
    return new_pop
