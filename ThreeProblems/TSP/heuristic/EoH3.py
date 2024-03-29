def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    max_n_used = np.max(edge_n_used)
    penalty_factor = 0.6 * (max_n_used - edge_n_used)**1.2

    for i in range(len(local_opt_tour)-1):
        edge_i = local_opt_tour[i]
        edge_j = local_opt_tour[i+1]

        updated_edge_distance[edge_i][edge_j] += penalty_factor[edge_i][edge_j]
        updated_edge_distance[edge_j][edge_i] = updated_edge_distance[edge_i][edge_j] # symmetrical matrix

    return updated_edge_distance