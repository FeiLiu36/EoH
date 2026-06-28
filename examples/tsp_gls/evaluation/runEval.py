import os
import sys
import time
import pickle

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation

_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'TestingData')

PROBLEM_SIZES = ['TSP20']
N_TEST = 16          # number of test instances per size
TIME_LIMIT = 10.0    # seconds of GLS per instance
ITE_MAX = 1000

print("TSP-GLS evaluation...")
with open("results.txt", "w") as fout:
    for name in PROBLEM_SIZES:
        with open(os.path.join(_TESTDATA_DIR, f'{name}.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        eva = Evaluation(dataset, N_TEST, time_limit=TIME_LIMIT, ite_max=ITE_MAX)
        t0 = time.time()
        avg_gap = eva.evaluate()
        result = (f"Avg optimality gap on {N_TEST} {name} instances: "
                  f"{avg_gap:7.4f}%   time: {time.time() - t0:7.1f}s")
        print(result)
        fout.write(result + "\n")
