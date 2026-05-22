import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation
from get_instance import GetData

N_JOBS = 20
N_MACHINES = 5
N_TEST = 16

print("FSSP-GLS evaluation...")
dataset = GetData(N_TEST, N_JOBS, N_MACHINES).generate_instances()
eva = Evaluation(dataset, N_TEST, N_JOBS, N_MACHINES, iter_max=200, time_max=15.0)

t0 = time.time()
avg_cmax = eva.evaluate()
result = (f"Avg makespan on {N_TEST} instances "
          f"({N_JOBS} jobs, {N_MACHINES} machines): {avg_cmax:.2f}  "
          f"time: {time.time()-t0:.1f}s")
print(result)

with open("results.txt", "w") as f:
    f.write(result + "\n")
