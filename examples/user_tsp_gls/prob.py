import numpy as np
import time
from joblib import Parallel, delayed
import os
import types
import warnings
import sys

from utils import readTSPRandom
from gls.gls_run import solve_instance

class TSPGLS():
    def __init__(self) -> None:
        self.n_inst_eva = 3 # a samll value for test only
        self.time_limit = 10 # maximum 10 seconds for each instance
        self.ite_max = 1000 # maximum number of local searchs in GLS for each instance
        self.perturbation_moves = 1 # movers of each edge in each perturbation
        path = os.path.dirname(os.path.abspath(__file__))
        self.instance_path = path+'/TrainingData/TSPAEL64.pkl' #,instances=None,instances_name=None,instances_scale=None
        self.debug_mode=False

        self.coords,self.instances,self.opt_costs = readTSPRandom.read_instance_all(self.instance_path)

        from prompts import GetPrompts
        self.prompts = GetPrompts()


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

    def evaluateGLS(self,heuristic):

        gaps = np.zeros(self.n_inst_eva)

        for i in range(self.n_inst_eva):
            gap = solve_instance(i,self.opt_costs[i],  
                                 self.instances[i], 
                                 self.coords[i],
                                 self.time_limit,
                                 self.ite_max,
                                 self.perturbation_moves,
                                 heuristic)
            gaps[i] = gap

        return np.mean(gaps)
    

    # def evaluateGLS(self,heuristic):

    #     nins = 64    
    #     gaps = np.zeros(nins)

    #     print("Start evaluation ...")   

    #     inputs = [(x,self.opt_costs[x],  self.instances[x], self.coords[x],self.time_limit,self.ite_max,self.perturbation_moves) for x in range(nins)]
    #     #gaps = Parallel(n_jobs=nins)(delayed(solve_instance)(*input) for input in inputs)
    #     try:
    #             gaps = Parallel(n_jobs= 4, timeout = self.time_limit*1.1)(delayed(solve_instance)(*input) for input in inputs)
    #     except:
    #             print("### timeout or other error, return a large fitness value ###")
    #             return 1E10
    #     return np.mean(gaps)


    # def evaluate(self):
    #     try:        
    #         fitness = self.evaluateGLS()
    #         return fitness
    #     except Exception as e:
    #         print("Error:", str(e))  # Print the error message
    #         return None

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

                #print(code_string)
                fitness = self.evaluateGLS(heuristic_module)

                return fitness
            
        except Exception as e:
            #print("Error:", str(e))
            return None



