# Template heuristic: Ant System (AS) pheromone update – classic baseline.
# Replace the body of `update_pheromone` with the best rule found by EoH.

import numpy as np


def update_pheromone(pheromone: np.ndarray, ant_tours: list, tour_costs: np.ndarray,
                     best_tour: np.ndarray, best_cost: float,
                     rho: float, iteration: int, max_iterations: int) -> np.ndarray:
    """Ant System: evaporate all edges, then all ants deposit proportional to 1/cost."""
    n = pheromone.shape[0]
    pheromone = (1.0 - rho) * pheromone
    for tour, cost in zip(ant_tours, tour_costs):
        delta = 1.0 / cost
        for i in range(n):
            u, v = int(tour[i]), int(tour[(i + 1) % n])
            pheromone[u, v] += delta
            pheromone[v, u] += delta
    return pheromone
