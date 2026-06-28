import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation

_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'TestingData', 'Taillard')

# Each Taillard file contains 10 instances. We evaluate a few standard sizes.
TEST_FILES = ['t_j20_m5.txt', 't_j20_m10.txt', 't_j50_m5.txt']
N_TEST = 5           # instances per file
TIME_MAX = 30.0      # seconds of GLS per instance
ITER_MAX = 1000


def read_taillard(path, n_test):
    """Parse a Taillard flow-shop file into (tasks_val, machines_val, tasks).

    Processing times are stored machine-major (one row per machine, one column
    per job); we transpose to tasks[job][machine].
    """
    lines = open(path).read().splitlines()
    instances = []
    i = 0
    while i < len(lines) and len(instances) < n_test:
        if 'number of jobs' in lines[i]:
            n_jobs, n_machines = (int(x) for x in lines[i + 1].split()[:2])
            rows = []
            for k in range(n_machines):
                rows.append([int(v) for v in lines[i + 3 + k].split()])
            tasks = np.array(rows, dtype=float).T   # -> (n_jobs, n_machines)
            instances.append((n_jobs, n_machines, tasks))
            i += 3 + n_machines
        else:
            i += 1
    return instances


print("FSSP-GLS evaluation...")
with open("results.txt", "w") as fout:
    for fname in TEST_FILES:
        instances = read_taillard(os.path.join(_TESTDATA_DIR, fname), N_TEST)
        eva = Evaluation(instances, time_max=TIME_MAX, iter_max=ITER_MAX)
        t0 = time.time()
        avg_cmax = eva.evaluate()
        result = (f"Avg makespan on {len(instances)} {fname} instances: "
                  f"{avg_cmax:9.2f}   time: {time.time() - t0:7.1f}s")
        print(result)
        fout.write(result + "\n")
