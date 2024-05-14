import numpy as np
import matplotlib.pyplot as plt
import importlib
import time


class Evaluation():
    def __init__(self,problem_size,dataset,n_test,debug_mode=False) -> None:

        self.ndelay = 1
        self.problem_size = problem_size
        self.neighbor_size = problem_size
        self.n_instance = n_test
        
        self.instance_data = dataset


    def tour_cost(self,instance, solution, problem_size):
        cost = 0
        for j in range(problem_size - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost

    def generate_neighborhood_matrix(self,instance):
        instance = np.array(instance)
        n = len(instance)
        neighborhood_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            distances = np.linalg.norm(instance[i] - instance, axis=1)
            sorted_indices = np.argsort(distances)  # sort indices based on distances
            neighborhood_matrix[i] = sorted_indices

        return neighborhood_matrix
    

    def route_plot(self, instance, route):
        LLM_route = np.array([instance[int(node_id)] for node_id in route])

        # Plot LLM_route and opt_route
        plt.figure(figsize=(8, 6))
        plt.plot(LLM_route[:, 0], LLM_route[:, 1], 'b-', linewidth=0.5, label='route')
        plt.scatter(LLM_route[:, 0], LLM_route[:, 1], c='b', s=10, label='nodes')

        # Display visited number above each node with larger font size
        for i, (x, y) in enumerate(LLM_route):
            plt.text(x, y, i + 1, ha='center', va='bottom', fontsize=6)

        # Add labels and legend with larger font size
        plt.xlabel('X coordinate', fontsize=14)
        plt.ylabel('Y coordinate', fontsize=14)
        plt.legend(fontsize=12)

        # Adjust the margin in the boundary
        plt.tight_layout()

        # Save the figure
        plt.savefig('route_plot.pdf')
        plt.savefig('route_plot.jpg')

        # Show the figure
        plt.show()


    def evaluate(self):
  
        heuristic_module = importlib.import_module("heuristic")
        eva = importlib.reload(heuristic_module)     

        dis = np.zeros(self.n_instance)
        n_ins = 0
        for instance, distance_matrix in self.instance_data:
            if n_ins == self.n_instance: break

            # get neighborhood matrix, we do not need it 
            neighbor_matrix = self.generate_neighborhood_matrix(instance)

            destination_node = 0

            current_node = 0

            route = np.zeros(self.problem_size)
            #print(">>> Step 0 : select node "+str(instance[0][0])+", "+str(instance[0][1]))
            for i in range(1,self.problem_size-1):

                near_nodes = neighbor_matrix[current_node][1:]

                mask = ~np.isin(near_nodes,route[:i])

                unvisited_near_nodes = near_nodes[mask]

                unvisited_near_size = np.minimum(self.neighbor_size,unvisited_near_nodes.size)

                unvisited_near_nodes = unvisited_near_nodes[:unvisited_near_size]

                next_node = eva.select_next_node(current_node, destination_node, unvisited_near_nodes, distance_matrix)

                current_node = next_node

                route[i] = current_node

                #print(">>> Step "+str(i)+": select node "+str(instance[current_node][0])+", "+str(instance[current_node][1]))

            mask = ~np.isin(np.arange(self.problem_size),route[:self.problem_size-1])

            last_node = np.arange(self.problem_size)[mask]

            current_node = last_node[0]

            route[self.problem_size-1] = current_node

            distance = self.tour_cost(instance,route,self.problem_size)

            dis[n_ins] = distance

            n_ins += 1


        ave_dis = np.average(dis)

        return ave_dis




