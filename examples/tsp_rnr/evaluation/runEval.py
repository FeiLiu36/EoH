import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation
from get_instance import GetData

N_NODES = 50
N_TEST = 16

print("TSP Ruin-and-Recreate evaluation...")
dataset = GetData(N_TEST, N_NODES).generate_instances()
eva = Evaluation(dataset, N_TEST, iter_max=200, time_max=10.0)

t0 = time.time()
avg_cost = eva.evaluate()
result = (f"Avg tour length on {N_TEST} instances, {N_NODES} nodes: "
          f"{avg_cost:.4f}  time: {time.time() - t0:.1f}s")
print(result)

with open("results.txt", "w") as f:
    f.write(result + "\n")
