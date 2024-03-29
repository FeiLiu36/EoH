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
    updated_edge_distance = np.copy(edge_distance)

    for i in range(num_nodes - 1):
        current_node = local_opt_tour[i]
        next_node = local_opt_tour[i + 1]

        if edge_n_used[current_node, next_node] > 0:
            updated_edge_distance[current_node, next_node] *= 1 + 0.05 * edge_n_used[current_node, next_node]
        else:
            if current_node % 2 == 0:
                updated_edge_distance[current_node, next_node] *= 1.5
            else:
                updated_edge_distance[current_node, next_node] *= 1.2

        if i % 2 == 0:
            updated_edge_distance[current_node, next_node] *= 1.25

    updated_edge_distance[local_opt_tour[-1], local_opt_tour[0]] *= edge_n_used[local_opt_tour[-1], local_opt_tour[0]] + 1

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and i % 2 == 1 and j % 2 == 0:
                updated_edge_distance[i, j] -= 5

    for i in range(num_nodes - 1):
        for j in range(i+1, num_nodes):
            if edge_n_used[i, j] == 0:
                updated_edge_distance[i, j] *= 1.5
            elif edge_n_used[i, j] <= 3:
                updated_edge_distance[i, j] *= 1.2
            else:
                updated_edge_distance[i, j] *= 1.1

    negative_edges = []
    for i in range(num_nodes - 1):
        for j in range(i+1, num_nodes):
            if updated_edge_distance[i, j] < 0:
                negative_edges.append((i, j))

    for edge in negative_edges:
        updated_edge_distance[edge] = 0

    updated_edge_distance = np.maximum(updated_edge_distance, 0)

    return updated_edge_distance