import numpy as np


class GetData:
    def __init__(self, n_instance, n_cities):
        self.n_instance = n_instance
        self.n_cities = n_cities

    def generate_instances(self):
        """Returns list of (coordinates, distance_matrix) tuples."""
        np.random.seed(2024)
        instances = []
        for _ in range(self.n_instance):
            coords = np.random.rand(self.n_cities, 2)
            dist = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
            instances.append((coords, dist))
        return instances
