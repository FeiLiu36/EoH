import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation

_TSPLIB_DIR = os.path.join(os.path.dirname(__file__), '..', 'TestingData', 'TSPLib')

TIME_LIMIT = 60.0    # seconds of GLS per instance
ITE_MAX = 1000
# Each GLS run keeps one core busy for TIME_LIMIT seconds; more workers than
# cores would starve the searches and degrade the reported gaps.
N_PROC = 8

# TSPLIB EUC_2D instances with 50-200 cities and their best-known (optimal)
# tour lengths, computed on integer-rounded distances (TSPLIB convention).
BKS = {
    'eil51': 426, 'berlin52': 7542, 'st70': 675, 'eil76': 538,
    'pr76': 108159, 'rat99': 1211, 'kroA100': 21282, 'kroB100': 22141,
    'kroC100': 20749, 'kroD100': 21294, 'kroE100': 22068, 'rd100': 7910,
    'eil101': 629, 'lin105': 14379, 'pr107': 44303, 'pr124': 59030,
    'bier127': 118282, 'ch130': 6110, 'pr136': 96772, 'pr144': 58537,
    'ch150': 6528, 'kroA150': 26524, 'kroB150': 26130, 'pr152': 73682,
    'u159': 42080, 'rat195': 2323, 'd198': 15780, 'kroA200': 29368,
    'kroB200': 29437,
}


def read_tsplib(path):
    """Read a TSPLIB EUC_2D instance and scale coordinates to [0, 1].

    Both axes are scaled jointly by (max - min) over all coordinate values so
    the geometry is preserved. Returns the scaled distance matrix and the
    scale factor, so a cost on the scaled instance times `scale` is comparable
    to the best-known solution on the original coordinates.
    """
    with open(path) as f:
        lines = f.read().splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.startswith('NODE_COORD_SECTION')) + 1
    coords = []
    for line in lines[start:]:
        parts = line.split()
        if not parts or parts[0] == 'EOF':
            break
        coords.append((float(parts[1]), float(parts[2])))
    coords = np.array(coords)

    cmin, cmax = coords.min(), coords.max()
    scale = cmax - cmin
    coords = (coords - cmin) / scale

    diff = coords[:, None, :] - coords[None, :, :]
    dis_matrix = np.sqrt((diff ** 2).sum(axis=-1))
    return dis_matrix, scale


def instance_size(name):
    return int(''.join(c for c in name if c.isdigit()))


def eval_instance(name):
    """Evaluate one TSPLIB instance; runs in a worker process."""
    dis_matrix, scale = read_tsplib(os.path.join(_TSPLIB_DIR, f'{name}.tsp'))
    # Express the best-known cost on the scaled instance so the gap
    # reported by Evaluation is the gap to the BKS.
    dataset = {'distance_matrix': [dis_matrix], 'cost': [BKS[name] / scale]}

    eva = Evaluation(dataset, 1, time_limit=TIME_LIMIT, ite_max=ITE_MAX)
    t0 = time.time()
    gap, iters_used = eva.evaluate()
    return name, dis_matrix.shape[0], gap, time.time() - t0, iters_used[0]


if __name__ == '__main__':
    names = sorted(BKS, key=instance_size)
    print(f"TSP-GLS evaluation on {len(names)} TSPLIB instances (50-200 cities), "
          f"coordinates scaled to [0,1], {N_PROC} parallel workers...")

    t_start = time.time()
    gaps = []
    with open("results.txt", "w") as fout:
        with Pool(processes=N_PROC) as pool:
            for name, n, gap, elapsed, iters_used in pool.imap(eval_instance, names):
                gaps.append(gap)
                cutoff = "TIME_LIMIT" if iters_used < ITE_MAX else "ite_max"
                result = (f"{name:<10s} (n={n:>3d})  "
                          f"gap to BKS: {gap:7.4f}%   time: {elapsed:6.1f}s   "
                          f"iters: {iters_used:>4d}/{ITE_MAX} (stopped: {cutoff})")
                print(result, flush=True)
                fout.write(result + "\n")

        summary = (f"Average gap on {len(gaps)} instances: {np.mean(gaps):7.4f}%   "
                   f"total time: {time.time() - t_start:6.1f}s")
        print(summary)
        fout.write(summary + "\n")
