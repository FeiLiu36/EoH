# Template: standard NSGA-II crowding distance (baseline).
# Replace the body of `crowding_distance` with the best operator found by EoH.

import numpy as np


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Standard NSGA-II crowding distance — classic baseline.

    Replace this with the best diversity metric discovered by EoH.
    """
    n, m = F.shape
    dist = np.zeros(n)
    for obj in range(m):
        idx = np.argsort(F[:, obj])
        dist[idx[0]] = np.inf
        dist[idx[-1]] = np.inf
        f_range = F[idx[-1], obj] - F[idx[0], obj]
        if f_range < 1e-10:
            continue
        for k in range(1, n - 1):
            dist[idx[k]] += (F[idx[k + 1], obj] - F[idx[k - 1], obj]) / f_range
    return dist
