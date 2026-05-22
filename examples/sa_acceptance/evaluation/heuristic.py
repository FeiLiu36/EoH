# Template heuristic: classic Boltzmann acceptance criterion – the SA baseline.
# Replace the body of `acceptance_probability` with the best function found by EoH.

import numpy as np


def acceptance_probability(delta_fitness: float, temperature: float,
                           iteration: int, max_iterations: int) -> float:
    import numpy as np
    progress = iteration / max_iterations
    # Threshold shifts from 2.0 to 0.5 over iterations
    threshold = 2.0 - 1.5 * progress
    # Steepness increases from 0.5 to 5.0
    steepness = 0.5 + 4.5 * progress
    # Logistic sigmoid centered at threshold
    z = steepness * (threshold - delta_fitness / max(temperature, 1e-10))
    prob = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(prob, 0.0, 1.0))
    