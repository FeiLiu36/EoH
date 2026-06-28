import os

import numpy as np


class GetData:
    """Loads flow-shop instances in Taillard format.

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
        tasks_val_list, machines_val_list, tasks_list = [], [], []
        for i in range(1, self.n_instance + 1):
            filename = os.path.join(self.data_dir, f"{i}.txt")
            tasks_val, machines_val, tasks = self._read_file(filename)
            tasks_val_list.append(tasks_val)
            machines_val_list.append(machines_val)
            tasks_list.append(tasks)
        return tasks_val_list, machines_val_list, tasks_list
