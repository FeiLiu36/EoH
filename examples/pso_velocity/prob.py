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


class PSOVelocityUpdate(BaseProblem):
    """EoH task: automatically design the velocity update rule for Particle Swarm Optimisation.

    The LLM designs `update_velocity`. The harness wraps it in a standard PSO
    loop (position update + bound-clipping) and evaluates it on five classic
    benchmark functions. Fitness is the mean log1p(best_found) across all
    benchmarks and random seeds — lower is better.
    """

    template_program = '''
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
    """Design a velocity update rule for Particle Swarm Optimisation (PSO).

    Args:
        velocities:      (pop_size, dim) current velocity vectors
        positions:       (pop_size, dim) current particle positions
        pbest_positions: (pop_size, dim) personal best positions found so far
        pbest_fitness:   (pop_size,) personal best objective values (lower = better)
        gbest_position:  (dim,) global best position found so far
        gbest_fitness:   global best objective value
        w:               inertia weight controlling momentum (default 0.729)
        c1:              cognitive coefficient — attraction toward personal best (default 1.494)
        c2:              social coefficient — attraction toward global best (default 1.494)
        bounds:          (dim, 2) each row is [lower_bound, upper_bound]
        iteration:       current iteration index (0-based)
        max_iterations:  total number of iterations planned

    Returns:
        new_velocities: (pop_size, dim) updated velocity vectors
    """
    pop_size, dim = velocities.shape
    r1 = np.random.rand(pop_size, dim)
    r2 = np.random.rand(pop_size, dim)
    cognitive = c1 * r1 * (pbest_positions - positions)
    social    = c2 * r2 * (gbest_position  - positions)
    return w * velocities + cognitive + social
'''

    task_description = (
        "Design a novel velocity update rule for the Particle Swarm Optimisation (PSO) "
        "algorithm. The function receives the current velocities, positions, personal "
        "best positions and fitnesses, the global best position and fitness, the inertia "
        "weight w, the cognitive coefficient c1, the social coefficient c2, the variable "
        "bounds, the current iteration index, and the maximum number of iterations. "
        "It must return an updated velocity array of the same shape (pop_size, dim). "
        "The standard PSO uses v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x), but "
        "you are encouraged to design more adaptive or creative strategies that exploit "
        "fitness information, iteration progress, or swarm diversity to improve "
        "convergence speed and solution quality. "
        "The goal is to minimise the average final objective value across a suite of "
        "10-dimensional continuous benchmark functions: Sphere, Rastrigin, Ackley, "
        "Rosenbrock, and Griewank."
    )

    def __init__(self, pop_size: int = 30, max_iterations: int = 200,
                 n_runs: int = 3, w: float = 0.729, c1: float = 1.494,
                 c2: float = 1.494, v_max_ratio: float = 0.2,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.n_runs = n_runs
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max_ratio = v_max_ratio
        self.instances = GetData().get_instances()

    def _run_pso(self, instance: dict, update_velocity_fn) -> float:
        """Run one PSO trial with the evolved velocity update rule."""
        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']
        bounds = np.column_stack([np.full(dim, lo), np.full(dim, hi)])
        v_max = self.v_max_ratio * (hi - lo)

        # Initialise swarm
        positions  = lo + (hi - lo) * np.random.rand(self.pop_size, dim)
        velocities = -v_max + 2 * v_max * np.random.rand(self.pop_size, dim)
        fitness    = np.array([func(p) for p in positions])

        pbest_positions = positions.copy()
        pbest_fitness   = fitness.copy()
        gbest_idx       = int(np.argmin(pbest_fitness))
        gbest_position  = pbest_positions[gbest_idx].copy()
        gbest_fitness   = float(pbest_fitness[gbest_idx])

        for iteration in range(self.max_iterations):
            new_vel = update_velocity_fn(
                velocities.copy(),
                positions.copy(),
                pbest_positions.copy(),
                pbest_fitness.copy(),
                gbest_position.copy(),
                gbest_fitness,
                self.w, self.c1, self.c2,
                bounds,
                iteration,
                self.max_iterations,
            )
            new_vel = np.asarray(new_vel, dtype=float)
            if new_vel.shape != (self.pop_size, dim):
                raise ValueError(
                    f"update_velocity returned shape {new_vel.shape}, "
                    f"expected ({self.pop_size}, {dim})"
                )

            # Clamp velocities
            velocities = np.clip(new_vel, -v_max, v_max)

            # Update positions
            positions = positions + velocities
            positions = np.clip(positions, lo, hi)

            # Evaluate and update personal/global bests
            fitness = np.array([func(p) for p in positions])
            improved = fitness < pbest_fitness
            pbest_positions[improved] = positions[improved].copy()
            pbest_fitness[improved]   = fitness[improved]

            best_idx = int(np.argmin(pbest_fitness))
            if pbest_fitness[best_idx] < gbest_fitness:
                gbest_fitness  = float(pbest_fitness[best_idx])
                gbest_position = pbest_positions[best_idx].copy()

        return gbest_fitness

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        scores = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                best = self._run_pso(instance, callable_func)
                run_bests.append(best)
            # log1p handles the wide dynamic range across benchmark functions
            scores.append(float(np.log1p(np.mean(run_bests))))
        return float(np.mean(scores))
