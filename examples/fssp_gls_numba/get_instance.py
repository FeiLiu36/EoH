import os
import re
from collections import namedtuple

import numpy as np


class GetData:
    """Loads flow-shop training instances.

    Each file starts with a line "<n_jobs> <n_machines>" followed by one line
    per job listing (machine_id, processing_time) pairs. The training set holds
    64 instances named 1.txt .. 64.txt.
    """

    def __init__(self, n_instance, data_dir=None):
        self.n_instance = n_instance
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'TrainingData')
        self.data_dir = data_dir

    def _read_file(self, filename):
        with open(filename, "r") as file:
            tasks_val, machines_val = file.readline().split()
            tasks_val = int(tasks_val)
            machines_val = int(machines_val)

            tasks = np.zeros((tasks_val, machines_val))
            for i in range(tasks_val):
                tmp = file.readline().split()
                for j in range(machines_val):
                    tasks[i][j] = int(float(tmp[j * 2 + 1]))
        return tasks_val, machines_val, tasks

    def load_instances(self):
        n_available = len([f for f in os.listdir(self.data_dir)
                           if f.endswith('.txt') and f[:-len('.txt')].isdigit()])
        if self.n_instance > n_available:
            raise ValueError(f"requested {self.n_instance} instances but only "
                             f"{n_available} are available in {self.data_dir}")

        tasks_val_list, machines_val_list, tasks_list = [], [], []
        for i in range(1, self.n_instance + 1):
            filename = os.path.join(self.data_dir, f"{i}.txt")
            tasks_val, machines_val, tasks = self._read_file(filename)
            tasks_val_list.append(tasks_val)
            machines_val_list.append(machines_val)
            tasks_list.append(tasks)
        return tasks_val_list, machines_val_list, tasks_list


Instance = namedtuple('Instance', 'name n_jobs n_machines ub lb tasks')
"""One Taillard instance. `ub`/`lb` are the best-known upper bound and the
lower bound shipped in the instance header; `tasks` is a C-contiguous float64
array of shape (n_jobs, n_machines)."""


class TaillardData:
    """Loads the Taillard flow-shop benchmark used for testing.

    TestingData/Taillard holds one file per problem size (t_j<n>_m<m>.txt),
    each containing 10 instances laid out as::

        number of jobs, number of machines, initial seed, upper bound and lower bound :
                  20           5   873654221        1278        1232
        processing times :
        <n_machines rows of n_jobs processing times>

    Processing times are stored machine-major, so they are transposed to
    tasks[job][machine] to match the training data and the GLS engine.
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__),
                                    'TestingData', 'Taillard')
        self.data_dir = data_dir

    def datasets(self):
        """Dataset file names, ordered by (n_jobs, n_machines)."""
        names = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]

        def size(name):
            m = re.match(r't_j(\d+)_m(\d+)\.txt$', name)
            return (int(m.group(1)), int(m.group(2))) if m else (1 << 30, name)

        return sorted(names, key=size)

    def load_dataset(self, name, n_test=None):
        """Parse one dataset file into a list of Instance tuples."""
        path = os.path.join(self.data_dir, name)
        with open(path) as f:
            lines = f.read().splitlines()

        instances = []
        i = 0
        while i < len(lines):
            if 'number of jobs' not in lines[i]:
                i += 1
                continue
            n_jobs, n_machines, _seed, ub, lb = (int(x) for x in lines[i + 1].split()[:5])
            rows = [[int(v) for v in lines[i + 3 + k].split()]
                    for k in range(n_machines)]
            # transpose to (n_jobs, n_machines); contiguous so the jitted local
            # search compiles a single specialisation
            tasks = np.ascontiguousarray(np.array(rows, dtype=float).T)
            instances.append(Instance(f"{name[:-4]}_{len(instances) + 1}",
                                      n_jobs, n_machines, ub, lb, tasks))
            i += 3 + n_machines
            if n_test is not None and len(instances) == n_test:
                break

        if not instances:
            raise ValueError(f"no instances parsed from {path}")
        return instances
