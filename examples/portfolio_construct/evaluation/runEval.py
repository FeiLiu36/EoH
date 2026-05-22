import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation
from get_instance import GetData

N_ASSETS = 20
N_SELECT = 5
N_PERIODS = 252
N_TEST = 16

print("Portfolio construction evaluation...")
dataset = GetData(N_TEST, N_ASSETS, N_PERIODS).generate_instances()
eva = Evaluation(dataset, N_TEST, n_select=N_SELECT)

t0 = time.time()
avg_sharpe = eva.evaluate()
result = (f"Avg annualised Sharpe on {N_TEST} instances, "
          f"{N_ASSETS} assets, select {N_SELECT}: "
          f"{avg_sharpe:.4f}  time: {time.time() - t0:.1f}s")
print(result)

with open("results.txt", "w") as f:
    f.write(result + "\n")
