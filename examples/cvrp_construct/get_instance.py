import numpy as np


class GetData:
    def __init__(self, n_instance, n_cities, capacity):
        self.n_instance = n_instance
        self.n_cities = n_cities   # includes depot at index 0
        self.capacity = capacity

    def generate_instances(self):
        """Returns list of (coordinates, distances, demands, capacity) tuples."""
        np.random.seed(2024)
        instances = []
        for _ in range(self.n_instance):
            coordinates = np.random.rand(self.n_cities, 2)
            demands = np.random.randint(1, 10, size=self.n_cities)
            demands[0] = 0  # depot has no demand
            distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
            instances.append((coordinates, distances, demands, self.capacity))
        return instances
