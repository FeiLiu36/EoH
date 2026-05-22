import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData


class Evaluation:
    """Post-hoc evaluator for PSO velocity update rules.

    Uses a larger budget and more seeds than the training evaluator in prob.py,
    and reports per-benchmark results alongside an overall summary.
    """

    # Extended benchmark suite: 10-D and 20-D variants
    CONFIGS = [
        {'name': 'sphere',     'dim': 10, 'bounds': (-5.12,   5.12)},
        {'name': 'sphere',     'dim': 20, 'bounds': (-5.12,   5.12)},
        {'name': 'rastrigin',  'dim': 10, 'bounds': (-5.12,   5.12)},
        {'name': 'rastrigin',  'dim': 20, 'bounds': (-5.12,   5.12)},
        {'name': 'ackley',     'dim': 10, 'bounds': (-32.768, 32.768)},
        {'name': 'ackley',     'dim': 20, 'bounds': (-32.768, 32.768)},
        {'name': 'rosenbrock', 'dim': 10, 'bounds': (-2.048,  2.048)},
        {'name': 'rosenbrock', 'dim': 20, 'bounds': (-2.048,  2.048)},
        {'name': 'griewank',   'dim': 10, 'bounds': (-600.0,  600.0)},
        {'name': 'griewank',   'dim': 20, 'bounds': (-600.0,  600.0)},
    ]

    def __init__(self, pop_size: int = 50, max_iterations: int = 500,
                 n_runs: int = 10, w: float = 0.729, c1: float = 1.494,
                 c2: float = 1.494, v_max_ratio: float = 0.2):
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.n_runs = n_runs
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max_ratio = v_max_ratio

        data = GetData()
        func_map = {
            'sphere':     data.sphere,
            'rastrigin':  data.rastrigin,
            'ackley':     data.ackley,
            'rosenbrock': data.rosenbrock,
            'griewank':   data.griewank,
        }
        self.instances = [{'func': func_map[c['name']], **c} for c in self.CONFIGS]

    def _run_pso(self, instance: dict, update_velocity_fn) -> float:
        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']
        bounds = np.column_stack([np.full(dim, lo), np.full(dim, hi)])
        v_max = self.v_max_ratio * (hi - lo)

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

            velocities = np.clip(new_vel, -v_max, v_max)
            positions  = np.clip(positions + velocities, lo, hi)

            fitness = np.array([func(p) for p in positions])
            improved = fitness < pbest_fitness
            pbest_positions[improved] = positions[improved].copy()
            pbest_fitness[improved]   = fitness[improved]

            best_idx = int(np.argmin(pbest_fitness))
            if pbest_fitness[best_idx] < gbest_fitness:
                gbest_fitness  = float(pbest_fitness[best_idx])
                gbest_position = pbest_positions[best_idx].copy()

        return gbest_fitness

    def evaluate(self, update_velocity_fn) -> list[dict]:
        """Evaluate update_velocity_fn on the full benchmark suite.

        Returns a list of result dicts, one per (function, dim) combination.
        Each dict has keys: name, dim, mean, std, log1p_mean.
        """
        results = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                np.random.seed(seed)
                best = self._run_pso(instance, update_velocity_fn)
                run_bests.append(best)
            results.append({
                'name':       instance['name'],
                'dim':        instance['dim'],
                'mean':       float(np.mean(run_bests)),
                'std':        float(np.std(run_bests)),
                'log1p_mean': float(np.log1p(np.mean(run_bests))),
            })
        return results
