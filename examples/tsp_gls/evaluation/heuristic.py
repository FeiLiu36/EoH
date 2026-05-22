import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Classic GLS penalty: augment tour edges proportional to d / (1 + times_penalised)."""
    aug = edge_distance.copy()
    n = len(local_opt_tour)
    for i in range(n):
        u = int(local_opt_tour[i])
        v = int(local_opt_tour[(i + 1) % n])
        delta = edge_distance[u, v] / (1.0 + edge_n_used[u, v])
        aug[u, v] += delta
        aug[v, u] += delta
    return aug
