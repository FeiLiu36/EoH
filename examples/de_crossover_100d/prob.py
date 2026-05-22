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


class DECrossover100D(BaseProblem):
    """EoH task: design the crossover operator for DE at 100 dimensions.

    Motivation
    ----------
    Standard binomial crossover treats all 100 dimensions identically.  In
    high-dimensional spaces this is wasteful: dimensions that are already
    well-adapted should be disturbed less, while stagnating dimensions need
    stronger injection from the mutant.  The LLM designs a smarter crossover
    rule that exploits generation progress, relative fitness, and any other
    signal derivable from the available arguments.

    Interface
    ---------
    Mutation is fixed as DE/rand/1:  v = x_r1 + F * (x_r2 - x_r3).
    The LLM designs `crossover`, which is called once per offspring to combine
    the target (current individual) with the mutant vector.

    Fitness: mean log1p(best_found) across all 100-D benchmarks and seeds
             (lower is better).
    """

    template_program = '''
def crossover(target: np.ndarray, mutant: np.ndarray, CR: float,
              generation: int, max_generations: int,
              fitness_target: float, fitness_best: float) -> np.ndarray:
    """Design a crossover operator for Differential Evolution.

    Args:
        target:          (dim,) current individual — the target vector
        mutant:          (dim,) mutant vector from DE/rand/1: v = x_r1 + F*(x_r2-x_r3)
        CR:              base crossover rate in (0, 1)
        generation:      current generation index (0-based)
        max_generations: total number of planned generations
        fitness_target:  objective value of target individual (lower is better)
        fitness_best:    best objective value in the current population

    Returns:
        trial: (dim,) trial vector combining elements from target and mutant
    """
    dim = len(target)
    mask = np.random.rand(dim) < CR
    mask[np.random.randint(dim)] = True  # ensure at least one element from mutant
    return np.where(mask, mutant, target)
'''

    task_description = (
        "Design a novel crossover (recombination) operator for Differential Evolution "
        "applied to 100-dimensional continuous black-box optimisation. "
        "The mutation step is fixed as DE/rand/1: v = x_r1 + F * (x_r2 - x_r3). "
        "The crossover function takes the target individual, the mutant vector, a base "
        "crossover rate CR, the current generation progress "
        "(generation index and total planned generations), "
        "and the objective values of the target and the population best. "
        "It must return a trial vector of the same dimensionality. "
        "The standard binomial crossover independently assigns each dimension from the "
        "mutant with probability CR, with at least one dimension forced from the mutant. "
        "You are encouraged to design more adaptive strategies: for example, "
        "progress-based adaptation (high CR for early exploration, low CR for late "
        "fine-tuning), fitness-aware selection (low-quality targets accept more mutant "
        "dimensions to drive exploration), exponential crossover that selects contiguous "
        "blocks of dimensions from the mutant, or dynamic strategies that change the "
        "effective recombination pressure based on convergence signals. "
        "The goal is to minimise the average final objective value across "
        "100-dimensional benchmark functions: Sphere, Rastrigin, Ackley, "
        "Rosenbrock, and Griewank."
    )

    def __init__(self, pop_size: int = 20, max_evals: int = 20000,
                 n_runs: int = 3, F: float = 0.8, CR: float = 0.9,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.n_runs = n_runs
        self.F = F
        self.CR = CR
        self.instances = GetData().get_instances(dim=100)

    def _run_de(self, instance: dict, crossover_fn) -> float:
        """Run one DE trial with fixed DE/rand/1 mutation and LLM-designed crossover."""
        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']
        bounds = np.column_stack([np.full(dim, lo), np.full(dim, hi)])
        max_gen = self.max_evals // self.pop_size

        pop = lo + (hi - lo) * np.random.rand(self.pop_size, dim)
        fitness = np.array([func(ind) for ind in pop])
        n_evals = self.pop_size
        best_idx = int(np.argmin(fitness))
        generation = 0

        while n_evals < self.max_evals:
            fitness_best = float(fitness[best_idx])

            for i in range(self.pop_size):
                if n_evals >= self.max_evals:
                    break

                # DE/rand/1 mutation (fixed)
                candidates = [j for j in range(self.pop_size) if j != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

                # LLM-designed crossover
                trial = crossover_fn(
                    pop[i].copy(), mutant, self.CR,
                    int(generation), int(max_gen),
                    float(fitness[i]), fitness_best,
                )
                trial = np.asarray(trial, dtype=float)
                if trial.shape != (dim,):
                    raise ValueError(
                        f"crossover returned shape {trial.shape}, expected ({dim},)"
                    )
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                # Greedy selection
                trial_fit = func(trial)
                n_evals += 1
                if trial_fit <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < fitness[best_idx]:
                        best_idx = i

            generation += 1

        return float(fitness[best_idx])

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        scores = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                best = self._run_de(instance, callable_func)
                run_bests.append(best)
            scores.append(float(np.log1p(np.mean(run_bests))))
        return float(np.mean(scores))
