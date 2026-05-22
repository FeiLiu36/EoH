# Template heuristic: standard PSO velocity update (cognitive + social).
# Replace the body of `update_velocity` with the best rule found by EoH.

import numpy as np


def update_velocity(
    velocities: np.ndarray,
    positions: np.ndarray,
    pbest_positions: np.ndarray,
    pbest_fitness: np.ndarray,
    gbest_position: np.ndarray,
    gbest_fitness: float,
    w: float,
    c1: float,
    c2: float,
    bounds: np.ndarray,
    iteration: int,
    max_iterations: int,
) -> np.ndarray:
    """Standard PSO: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)."""
    pop_size, dim = velocities.shape
    r1 = np.random.rand(pop_size, dim)
    r2 = np.random.rand(pop_size, dim)
    cognitive = c1 * r1 * (pbest_positions - positions)
    social    = c2 * r2 * (gbest_position  - positions)
    return w * velocities + cognitive + social
