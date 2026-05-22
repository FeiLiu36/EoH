import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation
from get_instance import GetData

# Test on larger set than training instances (different seed via larger n_instance)
N_CUSTOMERS = 50
CAPACITY = 40
N_TEST = 64

print("Start CVRP evaluation...")
dataset = GetData(N_TEST, N_CUSTOMERS + 1, CAPACITY).generate_instances()
eva = Evaluation(N_CUSTOMERS + 1, dataset, N_TEST, CAPACITY)

t0 = time.time()
avg_dist = eva.evaluate()
result = f"Avg distance on {N_TEST} instances, {N_CUSTOMERS} customers: {avg_dist:.4f}  time: {time.time()-t0:.3f}s"
print(result)

with open("results.txt", "w") as f:
    f.write(result + "\n")
