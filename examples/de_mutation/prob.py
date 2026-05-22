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


class DEMutation(BaseProblem):
    """EoH task: automatically design the mutation operator for Differential Evolution.

    The LLM designs the `mutation` function. The harness wraps it in a standard
    DE loop (binomial crossover + greedy selection) and evaluates it on five
    classic benchmark functions. Fitness is the mean log1p(best_found) across all
    benchmarks and random seeds — lower is better.
    """

    template_program = '''
def mutation(population: np.ndarray, current_idx: int, best_idx: int,
             fitness: np.ndarray, F: float, bounds: np.ndarray) -> np.ndarray:
    """Design a mutation operator for Differential Evolution.

    Args:
        population:  array of shape (pop_size, dim) – current population vectors
        current_idx: index of the target individual being evolved
        best_idx:    index of the best (lowest-fitness) individual in the population
        fitness:     array of shape (pop_size,) – objective values (lower = better)
        F:           scale factor in (0, 2) controlling the mutation step size
        bounds:      array of shape (dim, 2) – each row is [lower_bound, upper_bound]
    Returns:
        mutant: array of shape (dim,) – the mutant vector (before crossover/clipping)
    """
    pop_size, dim = population.shape
    candidates = [i for i in range(pop_size) if i != current_idx]
    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
    return population[r1] + F * (population[r2] - population[r3])
'''

    task_description = (
        "Design a novel mutation operator for the Differential Evolution (DE) "
        "optimisation algorithm. The mutation operator receives the current "
        "population, the index of the target individual, the index of the best "
        "individual, the fitness array, the scale factor F, and the variable "
        "bounds. It must return a mutant vector of the same dimensionality. "
        "Classic strategies include DE/rand/1, DE/best/1, and "
        "DE/current-to-best/1, but you are encouraged to design more adaptive "
        "or creative strategies that exploit fitness information or the "
        "population distribution. "
        "The goal is to minimise the average final objective value across a "
        "suite of 10-dimensional continuous benchmark functions: Sphere, "
        "Rastrigin, Ackley, Rosenbrock, and Griewank."
    )

    def __init__(self, pop_size: int = 20, max_evals: int = 5000,
                 n_runs: int = 3, F: float = 0.5, CR: float = 0.9,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.n_runs = n_runs
        self.F = F
        self.CR = CR
        self.instances = GetData().get_instances()

    def _run_de(self, instance: dict, mutation_fn) -> float:
        """Run one DE trial and return the best objective value found."""
        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']
        bounds = np.column_stack([np.full(dim, lo), np.full(dim, hi)])

        pop = lo + (hi - lo) * np.random.rand(self.pop_size, dim)
        fitness = np.array([func(ind) for ind in pop])
        n_evals = self.pop_size
        best_idx = int(np.argmin(fitness))

        while n_evals < self.max_evals:
            for i in range(self.pop_size):
                if n_evals >= self.max_evals:
                    break

                mutant = mutation_fn(
                    pop.copy(), i, best_idx, fitness.copy(), self.F, bounds
                )
                mutant = np.asarray(mutant, dtype=float)
                if mutant.shape != (dim,):
                    raise ValueError(f"mutant shape {mutant.shape} != ({dim},)")

                # Binomial crossover
                cross_mask = np.random.rand(dim) < self.CR
                cross_mask[np.random.randint(dim)] = True
                trial = np.where(cross_mask, mutant, pop[i])
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                # Greedy selection
                trial_fit = func(trial)
                n_evals += 1
                if trial_fit <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < fitness[best_idx]:
                        best_idx = i

        return float(fitness[best_idx])

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        scores = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                best = self._run_de(instance, callable_func)
                run_bests.append(best)
            # log1p handles the wide dynamic range across benchmark functions
            scores.append(float(np.log1p(np.mean(run_bests))))
        return float(np.mean(scores))
