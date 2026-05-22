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


class TSPCONST_MULTI(BaseProblem):
    """TSP construction via a two-function heuristic template.

    The LLM implements two cooperating functions:
      - compute_node_scores: assigns a priority score to each candidate node.
      - select_next_node: picks the best node using those scores.

    EoH receives select_next_node (the last top-level function) as callable_func.
    compute_node_scores is in the same exec namespace, so select_next_node can
    call it directly without any special wiring.
    """

    template_program = '''
def compute_node_scores(current_node: int, unvisited_nodes: np.ndarray,
                        distance_matrix: np.ndarray,
                        destination_node: int) -> np.ndarray:
    """Compute a priority score for each candidate unvisited node.

    Args:
        current_node:     index of the current node
        unvisited_nodes:  array of candidate unvisited node indices
        distance_matrix:  pairwise distances between all nodes
        destination_node: index of the final return node
    Returns:
        scores: 1-D array of scores, one per entry in unvisited_nodes
                (higher score = more preferred)
    """
    return -distance_matrix[current_node][unvisited_nodes]


def select_next_node(current_node: int, destination_node: int,
                     unvisited_nodes: np.ndarray,
                     distance_matrix: np.ndarray) -> int:
    """Select the next node using compute_node_scores.

    Args:
        current_node:     index of the current node
        destination_node: index of the final return node
        unvisited_nodes:  array of candidate unvisited node indices
        distance_matrix:  pairwise distances between all nodes
    Returns:
        Index of the next node to visit.
    """
    scores = compute_node_scores(current_node, unvisited_nodes,
                                 distance_matrix, destination_node)
    return unvisited_nodes[np.argmax(scores)]
'''

    task_description = (
        "Given a set of nodes with pairwise distances, design a two-function constructive "
        "heuristic for the Travelling Salesman Problem. "
        "compute_node_scores assigns a priority score to each candidate next node, and "
        "select_next_node uses those scores to pick the visit. "
        "Both functions together define the construction strategy; "
        "the goal is to minimise the total tour length."
    )

    def __init__(self, problem_size: int = 50, n_instance: int = 8,
                 timeout: int = 40, n_processes: int = 1):
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
        # callable_func is select_next_node (last top-level function).
        # compute_node_scores lives in the same exec namespace and is called
        # internally by select_next_node — no extra wiring needed here.
        distances = np.zeros(self.n_instance)
        for n_ins, (instance, distance_matrix) in enumerate(self.instance_data):
            neighbor_matrix = self._generate_neighborhood_matrix(instance)
            route = np.zeros(self.problem_size, dtype=int)
            current_node = 0
            destination_node = 0

            for i in range(1, self.problem_size - 1):
                near = neighbor_matrix[current_node][1:]
                unvisited = near[~np.isin(near, route[:i])]
                unvisited = unvisited[:min(self.neighbor_size, unvisited.size)]
                next_node = callable_func(
                    current_node, destination_node, unvisited, distance_matrix
                )
                if next_node in route[:i]:
                    return None
                current_node = int(next_node)
                route[i] = current_node

            remaining = np.arange(self.problem_size)[
                ~np.isin(np.arange(self.problem_size), route[:self.problem_size - 1])
            ]
            route[self.problem_size - 1] = remaining[0]
            distances[n_ins] = self._tour_cost(instance, route)

        return float(np.mean(distances))
