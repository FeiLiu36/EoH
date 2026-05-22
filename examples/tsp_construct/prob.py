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


class TSPCONST(BaseProblem):
    template_program = '''
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Select the next node to visit in a TSP greedy construction.

    Args:
        current_node: ID of the current node
        destination_node: ID of the destination (return) node
        unvisited_nodes: array of unvisited node IDs
        distance_matrix: pairwise distance matrix between all nodes
    Returns:
        next_node: ID of the next node to visit
    """
    return unvisited_nodes[np.argmin(distance_matrix[current_node][unvisited_nodes])]
'''
    task_description = (
        "Given a set of nodes with their coordinates, find the shortest route that visits each node once "
        "and returns to the starting node. The task is solved step-by-step by iteratively choosing the next node. "
        "Design a novel algorithm to select the next node in each step."
    )

    def __init__(self, problem_size=50, n_instance=8, timeout: int = 40, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.problem_size = problem_size
        self.neighbor_size = min(50, problem_size)
        self.n_instance = n_instance
        self.instance_data = GetData(n_instance, problem_size).generate_instances()

    def _tour_cost(self, instance, route):
        cost = sum(
            np.linalg.norm(instance[int(route[j])] - instance[int(route[j + 1])])
            for j in range(self.problem_size - 1)
        )
        cost += np.linalg.norm(instance[int(route[-1])] - instance[int(route[0])])
        return cost

    def _generate_neighborhood_matrix(self, instance):
        n = len(instance)
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            matrix[i] = np.argsort(np.linalg.norm(instance[i] - instance, axis=1))
        return matrix

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        distances = np.ones(self.n_instance)
        for n_ins, (instance, distance_matrix) in enumerate(self.instance_data):
            if n_ins == self.n_instance:
                break
            neighbor_matrix = self._generate_neighborhood_matrix(instance)
            route = np.zeros(self.problem_size)
            current_node = 0
            destination_node = 0
            for i in range(1, self.problem_size - 1):
                near = neighbor_matrix[current_node][1:]
                unvisited = near[~np.isin(near, route[:i])]
                unvisited = unvisited[:min(self.neighbor_size, unvisited.size)]
                next_node = callable_func(current_node, destination_node, unvisited, distance_matrix)
                if next_node in route:
                    return None
                current_node = next_node
                route[i] = current_node
            remaining = np.arange(self.problem_size)[~np.isin(np.arange(self.problem_size), route[:self.problem_size - 1])]
            route[self.problem_size - 1] = remaining[0]
            distances[n_ins] = self._tour_cost(instance, route)
        return float(np.mean(distances))
