def update_edge_distance(edge_distance: np.ndarray, local_opt_tour: np.ndarray, edge_n_used: np.ndarray) -> np.ndarray:
    num_nodes = edge_distance.shape[0]
    updated_edge_distance = np.copy(edge_distance)

    for i in range(num_nodes):
        current_node = local_opt_tour[i]
        next_node = local_opt_tour[(i + 1) % num_nodes]

        if current_node != next_node:
            updated_edge_distance[current_node, next_node] += edge_n_used[current_node, next_node]
            updated_edge_distance[next_node, current_node] += edge_n_used[next_node, current_node]

        if (updated_edge_distance[current_node, next_node] < 20 or
            updated_edge_distance[next_node, current_node] < 20):
            updated_edge_distance[current_node, next_node] += 5
            updated_edge_distance[next_node, current_node] += 5

        if (edge_n_used[current_node, next_node] > 5 or
            edge_n_used[next_node, current_node] > 5):
            updated_edge_distance[current_node, next_node] -= 5
            updated_edge_distance[next_node, current_node] -= 5

        if (updated_edge_distance[current_node, next_node] < 30 and
            updated_edge_distance[next_node, current_node] < 30):
            updated_edge_distance[current_node, next_node] += 8
            updated_edge_distance[next_node, current_node] += 8

        if (edge_n_used[current_node, next_node] > 8 or
            edge_n_used[next_node, current_node] > 8):
            updated_edge_distance[current_node, next_node] -= 5
            updated_edge_distance[next_node, current_node] -= 5

        if (updated_edge_distance[current_node, next_node] < 15 or
            updated_edge_distance[next_node, current_node] < 15):
            updated_edge_distance[current_node, next_node] = 15
            updated_edge_distance[next_node, current_node] = 15

        if (edge_n_used[current_node, next_node] > 15 or
            edge_n_used[next_node, current_node] > 15):
            updated_edge_distance[current_node, next_node] -= 10
            updated_edge_distance[next_node, current_node] -= 10

    return updated_edge_distance