import numpy as np
import networkx as nx
# Read coordinates from input file
from numba import jit
import os

#@jit(nopython=True) 
def read_coordinates(file_name):
    coordinates = []
    file = open(file_name, 'r')
    lines = file.readlines()
    for line in lines:
        if line.startswith('NODE_COORD_SECTION'):
            index = lines.index(line) + 1
            break
    for i in range(index, len(lines)-1):
        parts = lines[i].split()
        if (parts[0]=='EOF'): break
        coordinates.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return coordinates

#@jit(nopython=True) 
def create_distance_matrix(coordinates):

    x = np.array([coord[1] for coord in coordinates])
    y = np.array([coord[2] for coord in coordinates])

    min = np.min(coordinates)
    max = np.max(coordinates)

    x_normalized = (x - min) / (max - min)
    y_normalized = (y - min) / (max - min)

    x_diff = np.subtract.outer(x_normalized, x_normalized)
    y_diff = np.subtract.outer(y_normalized, y_normalized)
    distance_matrix = np.sqrt(x_diff**2 + y_diff**2)
    return distance_matrix, max - min

#@jit(nopython=True) 
def transform_to_graph(distance_matrix):
    num_nodes = distance_matrix.shape[0]

    graph = nx.Graph()

    for i in range(num_nodes):
        graph.add_node(i)

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            graph.add_edge(i, j, weight=distance_matrix[i, j], weight_name='weight')
            graph.add_edge(j, i, weight=distance_matrix[i, j], weight_name='weight')

    return graph

#@jit(nopython=True) 
def read_instance(filename):

    # Test the code
    coordinates = read_coordinates(filename)
    distance_matrix, scale = create_distance_matrix(coordinates)
    G = transform_to_graph(distance_matrix)

    # return distance matrix
    return G, scale

def read_instance_all(instances_path):
    instances = []
    instances_scale = []
    instances_name = []
    file_names = os.listdir(instances_path)
    for filename in file_names:
        G,scale = read_instance(instances_path +"/"+ filename)
        instances.append(G)
        instances_scale.append(scale)
        instances_name.append(filename)
    return instances,instances_scale,instances_name



if __name__ == '__main__':
    G,scale= read_instance_all('../TSPLib200/eil51.tsp')
    print(G.edges[0,0]['weight'])
    print(scale)