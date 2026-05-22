import os
import sys
import pickle
import time

sys.path.insert(0, os.path.dirname(__file__))

from evaluation import Evaluation

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'tsp_construct', 'testingdata')

problem_sizes = [20, 50, 100]
n_test = 64

print("Start evaluation (multi-function template)...")
with open("results.txt", "w") as f:
    for size in problem_sizes:
        data_path = os.path.join(TESTDATA_DIR, f"instance_data_{size}.pkl")
        with open(data_path, 'rb') as fp:
            dataset = pickle.load(fp)

        eva = Evaluation(size, dataset, n_test)
        t0 = time.time()
        avg_dist = eva.evaluate()
        result = (
            f"Avg distance on {n_test} instances, size {size}: "
            f"{avg_dist:.3f}  time: {time.time() - t0:.3f}s"
        )
        print(result)
        f.write(result + "\n")
