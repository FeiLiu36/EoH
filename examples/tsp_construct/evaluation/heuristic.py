# example heuristic
# replace it with your own heuristic designed by EoH
import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    threshold = 0.7
    scores = {}
    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance_to_unvisited = np.mean(all_distances) if len(all_distances) > 0 else 0.0
        std_dev_distance_to_unvisited = np.std(all_distances) if len(all_distances) > 0 else 0.0
        score = (0.4 * distance_matrix[current_node][node]
                 - 0.3 * average_distance_to_unvisited
                 + 0.2 * std_dev_distance_to_unvisited
                 - 0.1 * distance_matrix[destination_node][node])
        scores[node] = score
    if min(scores.values()) > threshold:
        next_node = min(unvisited_nodes, key=lambda node: distance_matrix[current_node][node])
    else:
        next_node = min(scores, key=scores.get)
    return next_node