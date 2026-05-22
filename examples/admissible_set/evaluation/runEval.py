import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from evaluation import Evaluation

DIMENSION = 15
WEIGHT = 10

print(f"Admissible set evaluation  I({DIMENSION}, {WEIGHT})")
eva = Evaluation(DIMENSION, WEIGHT)

t0 = time.time()
achieved, gap = eva.evaluate()
elapsed = time.time() - t0

result = (
    f"Set size: {achieved} / {eva.optimal_size} (optimal)  "
    f"gap: {gap:+d}  time: {elapsed:.2f}s"
)
print(result)

with open("results.txt", "w") as f:
    f.write(result + "\n")
