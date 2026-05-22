import importlib
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class Evaluation:
    def __init__(self, problem_size, dataset, n_test):
        self.problem_size = problem_size
        self.neighbor_size = problem_size
        self.n_instance = n_test
        self.instance_data = dataset

    def _tour_cost(self, instance, route):
        cost = sum(
            np.linalg.norm(instance[int(route[j])] - instance[int(route[j + 1])])
            for j in range(self.problem_size - 1)
        )
        cost += np.linalg.norm(instance[int(route[-1])] - instance[int(route[0])])
        return cost

    def _neighborhood_matrix(self, instance):
        n = len(instance)
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            matrix[i] = np.argsort(np.linalg.norm(instance[i] - instance, axis=1))
        return matrix

    def evaluate(self):
        # Reload so the file can be swapped between runs
        mod = importlib.reload(importlib.import_module("heuristic"))

        distances = np.zeros(self.n_instance)
        for n_ins, (instance, distance_matrix) in enumerate(self.instance_data):
            if n_ins == self.n_instance:
                break
            neighbor_matrix = self._neighborhood_matrix(instance)
            route = np.zeros(self.problem_size, dtype=int)
            current_node = 0
            destination_node = 0

            for i in range(1, self.problem_size - 1):
                near = neighbor_matrix[current_node][1:]
                unvisited = near[~np.isin(near, route[:i])]
                unvisited = unvisited[:min(self.neighbor_size, unvisited.size)]
                next_node = mod.select_next_node(
                    current_node, destination_node, unvisited, distance_matrix
                )
                current_node = int(next_node)
                route[i] = current_node

            remaining = np.arange(self.problem_size)[
                ~np.isin(np.arange(self.problem_size), route[:self.problem_size - 1])
            ]
            route[self.problem_size - 1] = remaining[0]
            distances[n_ins] = self._tour_cost(instance, route)

        return float(np.mean(distances))
