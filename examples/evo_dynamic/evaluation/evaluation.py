"""Post-hoc evaluator for dynamic-EA response strategies.

Uses a larger budget (more instances, more changes, larger dimensionality)
than the training evaluator in prob.py, and reports per-scenario breakdowns.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prob import _sphere_fitness, _ea_step
from get_instance import GetData

# Evaluation scenarios: (n_dims, n_changes, sigma_change, label)
SCENARIOS = [
    (10, 15, 0.3,  '10D slow'),
    (10, 15, 0.8,  '10D fast'),
    (20, 15, 0.5,  '20D medium'),
    (20, 15, 1.0,  '20D rapid'),
]


class Evaluation:
    def __init__(self, n_test: int = 16, pop_size: int = 30, k_iter: int = 50):
        self.n_test = n_test
        self.pop_size = pop_size
        self.k_iter = k_iter

    def _run(self, trajectory: list, bounds: np.ndarray, respond_fn) -> float:
        """Run EA on one instance; return mean per-environment tracking error."""
        lower, upper = bounds[0], bounds[1]
        n_dims = bounds.shape[1]
        population = np.random.uniform(lower, upper, (self.pop_size, n_dims))
        fitness = _sphere_fitness(population, trajectory[0])
        best_pos = population[int(np.argmax(fitness))].copy()
        total_error = 0.0

        for env_idx, optimum in enumerate(trajectory):
            if env_idx > 0:
                new_pop = respond_fn(population.copy(), fitness.copy(),
                                     best_pos.copy(), bounds.copy())
                population = np.clip(np.asarray(new_pop, dtype=float), lower, upper)
                fitness = _sphere_fitness(population, optimum)

            for _ in range(self.k_iter):
                population = _ea_step(population, fitness, bounds)
                fitness = _sphere_fitness(population, optimum)

            best_pos = population[int(np.argmax(fitness))].copy()
            total_error += float(np.linalg.norm(best_pos - optimum))

        return total_error / len(trajectory)

    def evaluate(self, respond_fn) -> list[dict]:
        """Evaluate respond_fn across all scenarios.

        Returns a list of result dicts with keys:
            label, n_dims, n_changes, sigma_change, mean_error, std_error.
        """
        results = []
        for n_dims, n_changes, sigma_change, label in SCENARIOS:
            bounds = np.array([[-5.0] * n_dims, [5.0] * n_dims])
            instances = GetData(self.n_test, n_dims, n_changes, sigma_change).generate_instances()
            errors = []
            for i, traj in enumerate(instances):
                np.random.seed(i)
                err = self._run(traj, bounds, respond_fn)
                errors.append(err)
            results.append({
                'label':       label,
                'n_dims':      n_dims,
                'n_changes':   n_changes,
                'sigma_change': sigma_change,
                'mean_error':  float(np.mean(errors)),
                'std_error':   float(np.std(errors)),
            })
        return results
