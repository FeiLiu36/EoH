# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


# ── EA primitives ─────────────────────────────────────────────────────────────

def _sphere_fitness(population: np.ndarray, optimum: np.ndarray) -> np.ndarray:
    """Fitness = negative squared Euclidean distance to optimum (higher = better)."""
    return -np.sum((population - optimum[np.newaxis, :]) ** 2, axis=1)


def _ea_step(population: np.ndarray, fitness: np.ndarray,
             bounds: np.ndarray, sigma: float = 0.3) -> np.ndarray:
    """One EA generation: tournament selection (k=3) + Gaussian mutation."""
    pop_size, n_dims = population.shape
    idx = np.random.randint(0, pop_size, (pop_size, 3))
    winners = idx[np.arange(pop_size), np.argmax(fitness[idx], axis=1)]
    offspring = population[winners] + np.random.normal(0.0, sigma, (pop_size, n_dims))
    return np.clip(offspring, bounds[0], bounds[1])


# ── Problem class ─────────────────────────────────────────────────────────────

class EvoDynamic(BaseProblem):
    """Dynamic Optimisation with Evolutionary Algorithm — response strategy design.

    The LLM designs respond_to_change, the key step called once each time
    the environment (objective function) changes.  The function receives the
    current population (adapted to the OLD landscape), its fitness on that
    old landscape, the best-known position, and the search bounds.  It must
    return an updated population to help the EA quickly re-adapt to the new
    landscape.

    EA loop (per environment):
      1. If env_idx > 0: call respond_to_change → new population.
      2. Run k_iter generations of tournament-selection + Gaussian mutation
         on the current (new) landscape.
      3. Record tracking error: distance of best individual to true optimum.

    Fitness: mean tracking error across all environments and instances
             (lower = better).
    """

    template_program = '''
def respond_to_change(population: np.ndarray, fitness: np.ndarray,
                      best_position: np.ndarray,
                      bounds: np.ndarray) -> np.ndarray:
    """Regenerate the population when the environment changes.

    Args:
        population:    (pop_size, n_dims) current population, optimised for
                       the old environment whose fitness is stale after the change
        fitness:       (pop_size,) fitness values on the OLD environment
                       (higher = better, i.e. closer to the old optimum)
        best_position: (n_dims,) best individual from the old environment
        bounds:        (2, n_dims) array; bounds[0] = lower, bounds[1] = upper
    Returns:
        new_population: (pop_size, n_dims) updated population for the new environment;
                        must have the same shape as population
    """
    # Default: hypermutation — perturb every individual with large Gaussian noise
    sigma = (bounds[1] - bounds[0]).mean() * 0.1
    new_pop = population + np.random.normal(0.0, sigma, population.shape)
    return np.clip(new_pop, bounds[0], bounds[1])
'''

    task_description = (
        "In dynamic optimisation, the objective function shifts periodically "
        "and an evolutionary algorithm must continuously track the moving optimum. "
        "Design the population response strategy that is called each time a "
        "change is detected. "
        "The function receives the current population (tuned to the OLD landscape), "
        "its fitness values on the old landscape, the best-known position, and the "
        "search bounds. It must return an updated population that balances "
        "memory (exploiting proximity to the old optimum) and diversity "
        "(exploring the new landscape). "
        "Classic approaches include hypermutation, random immigrants, memory-based "
        "reinitialization, and hybrid schemes. "
        "The goal is to minimise the mean tracking error — the Euclidean distance "
        "from the best-found individual to the true new optimum — after each change."
    )

    def __init__(self, n_dims: int = 10, n_instance: int = 5,
                 n_changes: int = 10, sigma_change: float = 0.5,
                 pop_size: int = 30, k_iter: int = 30,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_dims = n_dims
        self.n_instance = n_instance
        self.n_changes = n_changes
        self.pop_size = pop_size
        self.k_iter = k_iter
        self.bounds = np.array([[-5.0] * n_dims, [5.0] * n_dims])
        self.instance_data = GetData(
            n_instance, n_dims, n_changes, sigma_change
        ).generate_instances()

    def _run(self, trajectory: list, respond_fn) -> float | None:
        """Run the EA on one dynamic instance; return mean tracking error."""
        lower, upper = self.bounds[0], self.bounds[1]
        population = np.random.uniform(lower, upper, (self.pop_size, self.n_dims))
        fitness = _sphere_fitness(population, trajectory[0])
        best_pos = population[int(np.argmax(fitness))].copy()
        total_error = 0.0

        for env_idx, optimum in enumerate(trajectory):
            if env_idx > 0:
                new_pop = respond_fn(
                    population.copy(), fitness.copy(),
                    best_pos.copy(), self.bounds.copy()
                )
                new_pop = np.asarray(new_pop, dtype=float)
                if new_pop.shape != population.shape:
                    return None
                population = np.clip(new_pop, lower, upper)
                fitness = _sphere_fitness(population, optimum)

            # Optimise on current landscape for k_iter generations
            for _ in range(self.k_iter):
                offspring = _ea_step(population, fitness, self.bounds)
                population = offspring
                fitness = _sphere_fitness(population, optimum)

            best_pos = population[int(np.argmax(fitness))].copy()
            total_error += float(np.linalg.norm(best_pos - optimum))

        return total_error / len(trajectory)

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        np.random.seed(42)
        errors = []
        for trajectory in self.instance_data:
            err = self._run(trajectory, callable_func)
            if err is None:
                return None
            errors.append(err)
        return float(np.mean(errors))
