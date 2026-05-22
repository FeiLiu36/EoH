import importlib
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class Evaluation:
    def __init__(self, problem_size, dataset, n_test, capacity):
        self.problem_size = problem_size  # includes depot
        self.n_instance = n_test
        self.capacity = capacity
        self.instance_data = dataset

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
                route.append(0)
                current_load = 0
                current_node = 0
            else:
                route.append(int(next_node))
                current_load += int(demands[int(next_node)])
                unvisited.discard(int(next_node))
                current_node = int(next_node)

            feasible = np.array([n for n in all_customers
                                  if n in unvisited and current_load + demands[n] <= capacity])
            if unvisited and len(feasible) == 0:
                route.append(0)
                current_load = 0
                current_node = 0
                feasible = np.array(list(unvisited))

        if route[-1] != 0:
            route.append(0)
        return route

    def evaluate(self):
        mod = importlib.reload(importlib.import_module("heuristic"))
        distances = np.zeros(self.n_instance)
        for i, (instance, dist_matrix, demands, cap) in enumerate(self.instance_data):
            if i == self.n_instance:
                break
            route = self._route_construct(dist_matrix, demands, cap, mod.select_next_node)
            distances[i] = self._tour_cost(instance, route)
        return float(np.mean(distances))
