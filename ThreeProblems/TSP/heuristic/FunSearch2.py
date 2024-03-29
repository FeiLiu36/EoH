def update_edge_distance(edge_distance: np.ndarray, local_opt_tour: np.ndarray, edge_n_used: np.ndarray) -> np.ndarray:
    """
    Args:
        edge_distance (np.ndarray): Original edge distance matrix.
        local_opt_tour (np.ndarray): Local optimal solution path.
        edge_n_used (np.ndarray): Matrix representing the number of times each edge is used.
    Return:
        updated_edge_distance: updated score of each edge distance matrix.
    """
    num_nodes = edge_distance.shape[0]
    updated_edge_distance = np.zeros_like(edge_distance)

    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            current_node = local_opt_tour[i]
            next_node = local_opt_tour[j]
            if current_node % 2 == 0 and next_node % 2 == 1:
                if edge_n_used[current_node, next_node] >= 3:
                    updated_edge_distance[current_node, next_node] = edge_distance[current_node, next_node] * 0.9
                else:
                    updated_edge_distance[current_node, next_node] = edge_distance[current_node, next_node] * 1.1
            elif current_node % 2 == 1 and next_node % 2 == 0:
                if edge_n_used[current_node, next_node] >= 2:
                    updated_edge_distance[current_node, next_node] = edge_distance[current_node, next_node] * 0.8


    for i in range(num_nodes):
        for j in range(num_nodes):
            if edge_n_used[i, j] == 0 and i != j:
                if i + j >= num_nodes:
                    updated_edge_distance[i, j] = 1.5 * edge_distance[i, j]
                else:
                    updated_edge_distance[i, j] = 2.0 * edge_distance[i, j]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if edge_distance[i, j] > 100:
                updated_edge_distance[i, j] *= 0.8

    return updated_edge_distance