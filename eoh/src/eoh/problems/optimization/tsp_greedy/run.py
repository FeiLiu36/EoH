import numpy as np
import pickle
import sys
import types
import warnings
from .prompts import GetPrompts
from .get_instance import GetData

class TSPCONST():
    def __init__(self) -> None:
        # ABS_PATH = os.path.dirname(os.path.abspath(__file__))
        # sys.path.append(ABS_PATH)  # This is for finding all the modules
        # Construct the absolute path to the pickle file
        #pickle_file_path = os.path.join(ABS_PATH, 'instances.pkl')

        # with open("./instances.pkl" , 'rb') as f:
        #     self.instance_data = pickle.load(f)
        self.ndelay = 1
        self.problem_size = 50
        self.neighbor_size = np.minimum(50,self.problem_size)
        self.n_instance = 8  
        self.running_time = 10



        self.prompts = GetPrompts()

        getData = GetData(self.n_instance,self.problem_size)
        self.instance_data = getData.generate_instances()
        

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


    #@func_set_timeout(5)
    def greedy(self,eva):

        dis = np.ones(self.n_instance)
        n_ins = 0
        for instance, distance_matrix in self.instance_data:

            # get neighborhood matrix
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

                if next_node in route:
                    #print("wrong algorithm select duplicate node, retrying ...")
                    return None

                current_node = next_node

                route[i] = current_node

                #print(">>> Step "+str(i)+": select node "+str(instance[current_node][0])+", "+str(instance[current_node][1]))

            mask = ~np.isin(np.arange(self.problem_size),route[:self.problem_size-1])

            last_node = np.arange(self.problem_size)[mask]

            current_node = last_node[0]

            route[self.problem_size-1] = current_node

            #print(">>> Step "+str(self.problem_size-1)+": select node "+str(instance[current_node][0])+", "+str(instance[current_node][1]))

            LLM_dis = self.tour_cost(instance,route,self.problem_size)
            dis[n_ins] = LLM_dis

            n_ins += 1
            if n_ins == self.n_instance:
                break
            #self.route_plot(instance,route,self.oracle[n_ins])

        ave_dis = np.average(dis)
        #print("average dis: ",ave_dis)
        return ave_dis


    def evaluate(self, code_string):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")
                
                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                # Now you can use the module as you would any other
                fitness = self.greedy(heuristic_module)
                return fitness
        except Exception as e:
            #print("Error:", str(e))
            return None
        # try:
        #     heuristic_module = importlib.import_module("ael_alg")
        #     eva = importlib.reload(heuristic_module)   
        #     fitness = self.greedy(eva)
        #     return fitness
        # except Exception as e:
        #     print("Error:",str(e))
        #     return None
            


