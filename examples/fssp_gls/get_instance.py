import numpy as np


class GetData:
    def __init__(self, n_instance, n_jobs, n_machines):
        self.n_instance = n_instance
        self.n_jobs = n_jobs
        self.n_machines = n_machines

    def generate_instances(self):
        """Returns list of processing-time matrices, shape (n_jobs, n_machines)."""
        np.random.seed(2024)
        instances = []
        for _ in range(self.n_instance):
            tasks = np.random.randint(1, 100, size=(self.n_jobs, self.n_machines))
            instances.append(tasks.astype(float))
        return instances
