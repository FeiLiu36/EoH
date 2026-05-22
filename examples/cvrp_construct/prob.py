# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import copy
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


class CVRPCONST(BaseProblem):
    """Capacitated Vehicle Routing Problem — constructive heuristic.

    The LLM designs select_next_node, which picks the next customer to visit
    at each step of a greedy route-construction procedure.

    Fitness: average total travel distance across all training instances
             (lower is better).
    """

    template_program = '''
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray,
                     rest_capacity: float, demands: np.ndarray,
                     distance_matrix: np.ndarray) -> int:
    """Select the next node to visit in a CVRP greedy construction.

    Args:
        current_node:    index of the current node (0 = depot)
        depot:           index of the depot (always 0)
        unvisited_nodes: array of feasible unvisited customer indices
                         (already filtered to satisfy remaining capacity)
        rest_capacity:   remaining vehicle capacity
        demands:         demand of every node (index 0 = depot demand = 0)
        distance_matrix: pairwise Euclidean distance matrix
    Returns:
        Index of the next node to visit, or 0 to return to the depot early.
    """
    return unvisited_nodes[np.argmin(distance_matrix[current_node][unvisited_nodes])]
'''

    task_description = (
        "Given a set of customers with demands and a vehicle with fixed capacity, "
        "design a constructive heuristic for the Capacitated Vehicle Routing Problem (CVRP). "
        "At each step the heuristic selects the next customer to visit. "
        "When the vehicle cannot serve any remaining customer it returns to the depot "
        "and restarts with full capacity. "
        "The goal is to minimise the total travel distance across all routes."
    )

    def __init__(self, n_customers: int = 50, capacity: int = 40,
                 n_instance: int = 16, timeout: int = 40, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.problem_size = n_customers + 1   # +1 for depot at index 0
        self.capacity = capacity
        self.n_instance = n_instance
        self.instance_data = GetData(n_instance, self.problem_size, capacity).generate_instances()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _tour_cost(self, instance, route):
        return sum(
            np.linalg.norm(instance[int(route[j])] - instance[int(route[j + 1])])
            for j in range(len(route) - 1)
        )

    def _route_construct(self, distance_matrix, demands, capacity, heuristic):
        route = [0]
        current_load = 0
        current_node = 0

        unvisited = set(range(1, self.problem_size))
        all_customers = np.arange(1, self.problem_size)
        feasible = all_customers.copy()

        max_steps = self.problem_size * self.problem_size
        steps = 0
        while unvisited and steps < max_steps:
            steps += 1
            next_node = heuristic(
                current_node, 0, feasible,
                float(capacity - current_load),
                demands.copy(), distance_matrix.copy()
            )
            if next_node == 0:
                # voluntary return to depot
                route.append(0)
                current_load = 0
                current_node = 0
            else:
                route.append(int(next_node))
                current_load += int(demands[int(next_node)])
                unvisited.discard(int(next_node))
                current_node = int(next_node)

            # recompute feasible: unvisited customers that fit in remaining capacity
            feasible = np.array([n for n in all_customers
                                  if n in unvisited and current_load + demands[n] <= capacity])
            if unvisited and len(feasible) == 0:
                # force return to depot to free capacity
                route.append(0)
                current_load = 0
                current_node = 0
                feasible = np.array(list(unvisited))

        if unvisited:
            return None  # heuristic failed to visit all customers

        if route[-1] != 0:
            route.append(0)

        if len(set(route)) != self.problem_size:
            return None  # some customer never visited or visited twice

        return route

    # ── EoH interface ──────────────────────────────────────────────────────────

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        distances = np.zeros(self.n_instance)
        for i, (instance, dist_matrix, demands, cap) in enumerate(self.instance_data):
            route = self._route_construct(dist_matrix, demands, cap, callable_func)
            if route is None:
                return None
            distances[i] = self._tour_cost(instance, route)
        return float(np.mean(distances))
