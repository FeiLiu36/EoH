import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = edge_distance.copy()
    for i in range(len(local_opt_tour)-1):
        edge = (local_opt_tour[i], local_opt_tour[i+1])
        edge_n_used_normalized = edge_n_used[edge] / np.max(edge_n_used)
        edge_distance_increase = 5.0 + 0.3 / np.power(edge_n_used_normalized + 5, 0.7)  # Different parameter setting, increase factor of 0.3 with power normalization
        updated_edge_distance[edge] += edge_distance_increase
        updated_edge_distance[edge[::-1]] += edge_distance_increase  # Update the reverse edge as well
    return updated_edge_distance