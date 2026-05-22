import numpy as np


class TSPConstructor:
    """Nearest-neighbour baseline — replace with an EoH-designed heuristic."""

    def select_next_node(self, current_node, destination_node,
                         unvisited_nodes, distance_matrix):
        return unvisited_nodes[np.argmin(distance_matrix[current_node][unvisited_nodes])]
