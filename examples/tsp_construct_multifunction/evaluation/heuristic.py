import numpy as np


def compute_node_scores(current_node, unvisited_nodes, distance_matrix, destination_node):
    """Nearest-neighbour baseline — replace with an EoH-designed scoring function."""
    return -distance_matrix[current_node][unvisited_nodes]


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    scores = compute_node_scores(current_node, unvisited_nodes, distance_matrix, destination_node)
    return unvisited_nodes[np.argmax(scores)]
