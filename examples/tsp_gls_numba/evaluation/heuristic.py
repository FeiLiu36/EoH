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
#import numba as nb

# # @nb.jit(nopython=True, cache=True)
# def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
#     updated_edge_distance = np.copy(edge_distance)
#     edge_count = np.zeros_like(edge_distance)
#     for i in range(len(local_opt_tour) - 1):
#         start = local_opt_tour[i]
#         end = local_opt_tour[i + 1]
#         edge_count[start][end] += 1
#         edge_count[end][start] += 1
#         # penalize local optimal route
#     edge_n_used_max = np.max(edge_n_used)

#     # calculate the average edge used
#     decay_factor = 0.1 # decay fastor
#     mean_distance = np.mean(edge_distance)
#     # calculate the average distance
#     for i in range(edge_distance.shape[0]):
#         for j in range(edge_distance.shape[1]):
#             if edge_count[i][j] > 0:
#                 noise_factor = (np.random.uniform(0.7, 1.3) / edge_count[i][j]) + ( edge_distance[i][j] / mean_distance) - (0.3 / edge_n_used_max) * edge_n_used[i][j]
#                 # calculate a hybrid noise factor
#                 updated_edge_distance[i][j] += noise_factor * (1 + edge_count[i][j]) - decay_factor * updated_edge_distance[i][j]

#     return updated_edge_distance
