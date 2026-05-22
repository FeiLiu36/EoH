# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import sys
import os
import random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from deap import base, creator, tools, algorithms as deap_algorithms
from eoh import BaseProblem
from get_instance import GetData

# creator classes are module-level globals in DEAP — guard against double-registration
if not hasattr(creator, "FitnessMinGA"):
    creator.create("FitnessMinGA", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "IndividualGA"):
    creator.create("IndividualGA", list, fitness=creator.FitnessMinGA)


class EASimpleSelection(BaseProblem):
    """EoH task: automatically design the parent selection operator for DEAP's eaSimple.

    The LLM designs `select`, which maps a fitness array to k parent indices.
    The harness wires it into DEAP's toolbox and calls deap.algorithms.eaSimple
    with DEAP's built-in cxSimulatedBinaryBounded (η_c=15) and
    mutPolynomialBounded (η_m=20, indpb=1/dim) as fixed variation operators.
    Fitness is mean log1p(best_found) across all benchmarks and seeds — lower is better.
    """

    template_program = '''
import numpy as np

def select(
    fitnesses: np.ndarray,
    k: int,
    tournament_size: int,
) -> np.ndarray:
    """Design a parent selection operator for DEAP\'s eaSimple GA (minimisation).

    Args:
        fitnesses:       (pop_size,) objective values for each individual (lower = better)
        k:               number of parents to select (typically equal to pop_size)
        tournament_size: reference tournament size supplied by the harness (default 3);
                         you may use it or ignore it in favour of your own strategy

    Returns:
        selected: (k,) integer array of parent indices (sampling with replacement allowed)
    """
    pop_size = len(fitnesses)
    selected = np.empty(k, dtype=int)
    for i in range(k):
        candidates = np.random.choice(pop_size, tournament_size, replace=False)
        selected[i] = candidates[np.argmin(fitnesses[candidates])]
    return selected
'''

    task_description = (
        "Design a novel parent selection operator for DEAP's eaSimple genetic algorithm "
        "(minimisation). The selection function receives the current population's fitness "
        "array (lower = better), the number of parents k to select, and a reference "
        "tournament size. It must return a length-k integer array of parent indices. "
        "The standard strategy is tournament selection (randomly sample tournament_size "
        "individuals, pick the best), but you are encouraged to design more adaptive "
        "or creative strategies — for example rank-based selection, Boltzmann/softmax "
        "selection with an annealing temperature, fitness-proportionate (roulette wheel) "
        "selection, or any hybrid that balances exploration and exploitation. "
        "The variation step (DEAP's SBX crossover + polynomial mutation) and full "
        "generational replacement are fixed by deap.algorithms.eaSimple; only the "
        "selection operator is evolved. "
        "The goal is to minimise the average final objective value across a suite of "
        "10-dimensional continuous benchmark functions: Sphere, Rastrigin, Ackley, "
        "Rosenbrock, and Griewank."
    )

    def __init__(self, pop_size: int = 50, n_gen: int = 100,
                 tournament_size: int = 3, cxpb: float = 0.9,
                 mutpb: float = 0.1, eta_c: float = 15.0, eta_m: float = 20.0,
                 n_runs: int = 3, timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.tournament_size = tournament_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.n_runs = n_runs
        self.instances = GetData().get_instances()

    def _run_eaSimple(self, instance: dict, select_fn, seed: int) -> float:
        """Run one DEAP eaSimple trial with the evolved selection operator."""
        random.seed(seed)       # DEAP's crossover/mutation use Python random
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

        # Bridge: DEAP select(individuals, k) → wrap the numpy-based evolved function
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

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        scores = []
        for instance in self.instances:
            run_bests = []
            for seed in range(self.n_runs):
                best = self._run_eaSimple(instance, callable_func, seed)
                run_bests.append(best)
            scores.append(float(np.log1p(np.mean(run_bests))))
        return float(np.mean(scores))
