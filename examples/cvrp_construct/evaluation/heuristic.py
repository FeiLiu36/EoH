import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """Nearest-feasible-neighbour baseline — replace with an EoH-designed heuristic."""
    return unvisited_nodes[np.argmin(distance_matrix[current_node][unvisited_nodes])]
