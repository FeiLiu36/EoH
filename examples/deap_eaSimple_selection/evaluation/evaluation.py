import sys
import os
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from deap import base, creator, tools, algorithms as deap_algorithms
from get_instance import GetData

# Re-use the same creator classes registered in prob.py; guard if loaded standalone
if not hasattr(creator, "FitnessMinGA"):
    creator.create("FitnessMinGA", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "IndividualGA"):
    creator.create("IndividualGA", list, fitness=creator.FitnessMinGA)


class Evaluation:
    """Post-hoc evaluator for eaSimple parent selection operators.

    Uses a larger population, more generations, and more seeds than the
    training evaluator in prob.py, and reports per-benchmark results
    alongside an overall summary. Runs DEAP's eaSimple identically to
    the training harness so scores are directly comparable.
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

    def __init__(self, pop_size: int = 100, n_gen: int = 200,
                 tournament_size: int = 3, cxpb: float = 0.9,
                 mutpb: float = 0.1, eta_c: float = 15.0,
                 eta_m: float = 20.0, n_runs: int = 10):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.tournament_size = tournament_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.n_runs = n_runs

        data = GetData()
        func_map = {
            'sphere':     data.sphere,
            'rastrigin':  data.rastrigin,
            'ackley':     data.ackley,
            'rosenbrock': data.rosenbrock,
            'griewank':   data.griewank,
        }
        self.instances = [{'func': func_map[c['name']], **c} for c in self.CONFIGS]

    def _run_eaSimple(self, instance: dict, select_fn, seed: int) -> float:
        random.seed(seed)
        np.random.seed(seed)

        func = instance['func']
        dim = instance['dim']
        lo, hi = instance['bounds']
        indpb = 1.0 / dim

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, lo, hi)
        toolbox.register("individual", tools.initRepeat,
                         creator.IndividualGA, toolbox.attr_float, n=dim)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", lambda ind: (func(np.array(ind)),))
        toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                         low=lo, up=hi, eta=self.eta_c)
        toolbox.register("mutate", tools.mutPolynomialBounded,
                         low=lo, up=hi, eta=self.eta_m, indpb=indpb)

        tournament_size = self.tournament_size

        def deap_select(individuals, k):
            fitnesses = np.array([ind.fitness.values[0] for ind in individuals])
            indices = np.asarray(select_fn(fitnesses, k, tournament_size), dtype=int)
            if indices.shape != (k,):
                raise ValueError(
                    f"select returned shape {indices.shape}, expected ({k},)")
            if not np.all((indices >= 0) & (indices < len(individuals))):
                raise ValueError("select returned out-of-range indices")
            return [individuals[i] for i in indices]

        toolbox.register("select", deap_select)

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)

        deap_algorithms.eaSimple(
            pop, toolbox,
            cxpb=self.cxpb, mutpb=self.mutpb,
            ngen=self.n_gen, halloffame=hof, verbose=False,
        )

        return float(hof[0].fitness.values[0])

    def evaluate(self, select_fn) -> list[dict]:
        """Evaluate select_fn on the full benchmark suite.

        Returns a list of result dicts, one per (function, dim) combination.
        Each dict has keys: name, dim, mean, std, log1p_mean.
        """
        results = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                best = self._run_eaSimple(instance, select_fn, seed)
                run_bests.append(best)
            results.append({
                'name':       instance['name'],
                'dim':        instance['dim'],
                'mean':       float(np.mean(run_bests)),
                'std':        float(np.std(run_bests)),
                'log1p_mean': float(np.log1p(np.mean(run_bests))),
            })
        return results
