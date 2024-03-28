import numpy as np
import time
from joblib import Parallel, delayed
import os

from utils import readTSPLib
from gls.gls_run import solve_instance_tsplib

class Evaluation():
    def __init__(self) -> None:
        self.time_limit = 60 # maximum 10 seconds for each instance
        self.ite_max = 1000 # maximum number of local searchs in GLS for each instance
        self.perturbation_moves = 1 # movers of each edge in each perturbation
        path = os.path.dirname(os.path.abspath(__file__))
        #self.instance_path = path+'/instance/TSPAEL64.pkl' #,instances=None,instances_name=None,instances_scale=None
        self.debug_mode=False

        #self.coords,self.instances,self.opt_costs = readTSPLib.read_instance_all(self.instance_path)
        m = 50 # neighborhood size
        instances_path = path+"/../instances/TSPLib200"
        coords = []
        instances = []
        neighbors = []
        instances_scale = []
        instances_name = []
        file_names = os.listdir(instances_path)
        for filename in file_names:
            coordinates,distance_matrix,neighbor_matrix,scale = readTSPLib.readinstance(instances_path +"/"+ filename,m)
            coords.append(coordinates)
            instances.append(distance_matrix)
            neighbors.append(neighbor_matrix)
            instances_scale.append(scale)
            instances_name.append(filename)
            
        self.coords,self.instances,self.instances_name,self.instances_scale = coords,instances,instances_name,instances_scale
    
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

    def evaluateGLS(self):

        time.sleep(1)

        nins = len(self.instances_name) 
        gaps = np.zeros(nins)

        print("Start evaluation ...")   

        inputs = [(x,self.instances_name[x],self.instances_scale[x],  self.instances[x], self.coords[x],self.time_limit,self.ite_max,self.perturbation_moves) for x in range(nins)]
        #gaps = Parallel(n_jobs=nins)(delayed(solve_instance)(*input) for input in inputs)

        gaps = Parallel(n_jobs= 4, timeout = self.time_limit*1.1)(delayed(solve_instance_tsplib)(*input) for input in inputs)

        if len(gaps) >= 50 and len(gaps) % 50 == 0:
            avg_gap = np.mean(gaps)
            print("Average Gap for", len(gaps), "instances:", avg_gap)
            
        print("Average Gap: ",np.mean(gaps))
        print("Evaluation finished !")
        input()

        return np.mean(gaps)


    def evaluate(self):
      
        fitness = self.evaluateGLS()
        return fitness



if __name__ == "__main__":
     eva = Evaluation()
     eva.evaluate()

