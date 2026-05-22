import numpy as np


def destroy_nodes(current_tour: np.ndarray, distance_matrix: np.ndarray,
                  n_destroy: int) -> np.ndarray:
    n = len(current_tour)
    # Randomly select a contiguous segment of the tour
    start = np.random.randint(0, n)
    end = (start + np.random.randint(1, n)) % n
    if start < end:
        segment_indices = np.arange(start, end)
    else:
        segment_indices = np.concatenate([np.arange(start, n), np.arange(0, end)])
    
    if len(segment_indices) < n_destroy:
        segment_indices = np.arange(n)
    
    # For each node in the segment, compute detour cost
    detour_costs = np.zeros(len(segment_indices))
    for i, idx in enumerate(segment_indices):
        prev = current_tour[(idx - 1) % n]
        curr = current_tour[idx]
        next_node = current_tour[(idx + 1) % n]
        # Distance of direct edge between neighbors
        direct_dist = distance_matrix[prev, next_node]
        # Sum of incident edges
        incident_sum = distance_matrix[prev, curr] + distance_matrix[curr, next_node]
        detour_costs[i] = incident_sum - direct_dist
    
    # Select the nodes with the largest detour cost within the segment
    top_local_indices = np.argsort(detour_costs)[-n_destroy:]
    nodes_to_remove = current_tour[segment_indices[top_local_indices]]
    return nodes_to_remove

