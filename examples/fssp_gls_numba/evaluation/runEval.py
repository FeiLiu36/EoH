"""Evaluate heuristic.py on the full Taillard flow-shop benchmark.

Runs GLS on every instance of all 11 datasets in TestingData/Taillard (110
instances, 20-200 jobs x 5-20 machines) and reports the average relative
makespan gap per dataset, against the best-known upper bound that ships in
each instance header.

On the larger datasets the per-instance budget (TIME_MAX) rather than ITER_MAX
stops the search — visible in the s/inst column — so their gaps move by a few
hundredths of a percent between runs even though GLS itself is seeded.
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation
from get_instance import TaillardData

TIME_MAX = 60.0      # seconds of GLS per instance
ITER_MAX = 1000
N_TEST = 10          # instances per dataset (each Taillard file holds 10)
# Each GLS run keeps one core busy for TIME_MAX seconds; more workers than
# cores would starve the searches and degrade the reported gaps.
N_PROC = 8


def eval_instance(inst):
    """Evaluate one instance; runs in a worker process."""
    eva = Evaluation([inst], time_max=TIME_MAX, iter_max=ITER_MAX)
    t0 = time.time()
    res = eva.evaluate()
    return res['gaps'][0], res['makespans'][0], time.time() - t0


if __name__ == '__main__':
    data = TaillardData()
    datasets = data.datasets()

    header = (f"FSSP-GLS evaluation on {len(datasets)} Taillard datasets, "
              f"{N_TEST} instances each, {TIME_MAX:g}s / {ITER_MAX} iters per "
              f"instance, {N_PROC} parallel workers\n"
              f"gap = (makespan / best-known upper bound - 1) * 100%, averaged "
              f"per dataset (lower = better)")
    columns = (f"{'dataset':<12s} {'jobs':>5s} {'mach':>5s} {'n':>4s} "
               f"{'avg makespan':>13s} {'avg gap to UB':>14s} {'s/inst':>8s}")

    t_start = time.time()
    all_gaps = []
    n_failed_total = 0

    with open("results.txt", "w") as fout:
        for line in (header, columns):
            print(line, flush=True)
            fout.write(line + "\n")

        with Pool(processes=N_PROC) as pool:
            for name in datasets:
                instances = data.load_dataset(name, N_TEST)
                t0 = time.time()
                results = pool.map(eval_instance, instances)

                gaps = [g for g, _c, _t in results if g is not None]
                cmaxs = [c for _g, c, _t in results if c is not None]
                n_failed = len(results) - len(gaps)
                all_gaps += gaps
                n_failed_total += n_failed

                # Mean per-instance solve time: at the TIME_MAX cap the search
                # was stopped by the clock, below it by ITER_MAX.
                row = (f"{name[:-4]:<12s} {instances[0].n_jobs:>5d} "
                       f"{instances[0].n_machines:>5d} {len(instances):>4d} "
                       f"{np.mean(cmaxs) if cmaxs else float('nan'):>13.2f} "
                       f"{np.mean(gaps) if gaps else float('nan'):>13.3f}% "
                       f"{np.mean([t for _g, _c, t in results]):>7.1f}s"
                       + (f"   ({n_failed} failed)" if n_failed else ""))
                print(row, flush=True)
                fout.write(row + "\n")

        summary = (f"{'all':<12s} {'':>5s} {'':>5s} {len(all_gaps):>4d} "
                   f"{'':>13s} "
                   f"{np.mean(all_gaps) if all_gaps else float('nan'):>13.3f}% "
                   f"   (total wall time {time.time() - t_start:.1f}s)"
                   + (f"   ({n_failed_total} failed)" if n_failed_total else ""))
        print(summary)
        fout.write(summary + "\n")
