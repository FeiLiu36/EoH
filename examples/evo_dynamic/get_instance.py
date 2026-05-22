import numpy as np


class GetData:
    """Generate dynamic optimisation instances with a moving global optimum.

    Each instance is a sequence of optimum positions that the EA must track.
    The optimum performs a bounded random walk in n_dims-dimensional space.

    The training suite pairs small-shift (slow change) and large-shift
    (rapid change) scenarios so that evolved response strategies generalise
    across different change severities.
    """

    def __init__(self, n_instance: int, n_dims: int = 10,
                 n_changes: int = 10, sigma_change: float = 0.5):
        self.n_instance = n_instance
        self.n_dims = n_dims
        self.n_changes = n_changes
        self.sigma_change = sigma_change

    def generate_instances(self):
        """Return list of optimum trajectories.

        Each element is a list of n_changes arrays of shape (n_dims,).
        Optima are kept within [-4, 4]^n_dims so the EA bounds [-5, 5]
        always bracket the true optimum.
        """
        np.random.seed(2024)
        instances = []
        for _ in range(self.n_instance):
            optimum = np.random.uniform(-3.0, 3.0, self.n_dims)
            trajectory = [optimum.copy()]
            for _ in range(self.n_changes - 1):
                step = np.random.normal(0.0, self.sigma_change, self.n_dims)
                optimum = np.clip(optimum + step, -4.0, 4.0)
                trajectory.append(optimum.copy())
            instances.append(trajectory)
        return instances
