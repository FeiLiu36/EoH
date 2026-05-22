import numpy as np


class GetData:
    """Random Euclidean TSP instance generator."""

    @staticmethod
    def generate_instances(n_instances: int = 5, n_nodes: int = 20,
                           seed: int = 42) -> list[dict]:
        """Return a list of TSP instance dicts with keys: coords, dist, n_nodes."""
        rng = np.random.RandomState(seed)
        instances = []
        for _ in range(n_instances):
            coords = rng.rand(n_nodes, 2) * 100.0
            diff = coords[:, None, :] - coords[None, :, :]
            dist = np.sqrt((diff ** 2).sum(-1))
            instances.append({'coords': coords, 'dist': dist, 'n_nodes': n_nodes})
        return instances
