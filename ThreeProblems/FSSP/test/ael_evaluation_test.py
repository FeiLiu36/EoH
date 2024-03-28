import numpy as np
import importlib
import time
from numba import jit
from joblib import Parallel, delayed
import ael_alg as alg
import random

@jit(nopython=True)
def makespan(order, tasks, machines_val):
    times = []
    for i in range(0, machines_val):
        times.append(0)
    for j in order:
        times[0] += tasks[j][0]
        for k in range(1, machines_val):
            if times[k] < times[k-1]:
                times[k] = times[k-1]
            times[k] += tasks[j][k]
    return max(times)


@jit(nopython=True)
def local_search(sequence, cmax_old,tasks,machines_val):
    new_seq = sequence[:]
    for i in range(len(new_seq)):
        for j in range(i+1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:  # Assuming cmax_old is defined in sim_ann1 function
                new_seq = temp_seq[:]
                cmax_old = cmax
                #print(cmax)

    for i in range(1,len(new_seq)):
        for j in range(1,len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:  # Assuming cmax_old is defined in sim_ann1 function
                new_seq = temp_seq[:]
                cmax_old = cmax
                #print(cmax)

    return new_seq

@jit(nopython=True)
def local_search_perturb(sequence, cmax_old,tasks,machines_val,job):
    new_seq = sequence[:]
    for i in job:
        for j in range(i+1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:  # Assuming cmax_old is defined in sim_ann1 function
                new_seq = temp_seq[:]
                cmax_old = cmax
                #print(cmax)

    for i in job:
        for j in range(1,len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:  # Assuming cmax_old is defined in sim_ann1 function
                new_seq = temp_seq[:]
                cmax_old = cmax
                #print(cmax)

    return new_seq



def perturbation_operator(sequence):
    n = len(sequence)

    # Kick-move (composed of two swap-moves and one interchange move)
    # Perform two swap-moves at randomly chosen positions
    for _ in range(2):
        index1 = random.randint(0, n-2)  # randomly choose first position
        index2 = index1 + 1              # second position is next neighbor
        sequence[index1], sequence[index2] = sequence[index2], sequence[index1]  # swap

    # Find two nodes with absolute distance less than max(n/5, 30) and exchange them
    distance_limit = max(n//5, 30)
    indices = random.sample(range(n), 2)  # randomly sample two indices
    index1, index2 = indices[0], indices[1]
    while abs(index1 - index2) >= distance_limit:  # ensure the absolute distance is less than the limit
        indices = random.sample(range(n), 2)
        index1, index2 = indices[0], indices[1]
    sequence[index1], sequence[index2] = sequence[index2], sequence[index1]  # swap

    return sequence
    
class Evaluation():
    def __init__(self,method,nc,iter_max,time_max,n,m,tasks,names) -> None:
        self.iter_max = iter_max
        self.time_max = time_max
        self.nc = nc
        self.tasks_val, self.machines_val, self.tasks, self.names = n,m,tasks,names
        self.method = method



    ############################################### Local Search ####################################################
    def ls(self,tasks_val, tasks, machines_val):
        pi0, cmax0 = self.neh(tasks, machines_val, tasks_val) 
        #print("neh results: ",cmax0) 
        #piprim = local_search(pi, tasks_val)
        pi = local_search(pi0, cmax0,tasks,machines_val)
        cmax = makespan(pi, tasks, machines_val)
        return cmax

    ############################################### AEL guided Local Search ####################################################
    def gls(self,tasks_val, tasks, machines_val):

        cmax_best = 1E10
        #print("run ...")
        try:
            pi, cmax = self.neh(tasks, machines_val, tasks_val) 
            n = len(pi)
            
            pi_best = pi
            cmax_best = cmax
            n_itr = 0
            time_start = time.time()
            while time.time() - time_start < self.time_max and n_itr <self.iter_max:
                #piprim = local_search(pi, tasks_val)
                piprim = local_search(pi, cmax,tasks,machines_val)

                pi = piprim
                cmax = makespan(pi, tasks, machines_val)
                
                if (cmax<cmax_best):
                    pi_best = pi
                    cmax_best = cmax

                tasks_perturb, jobs = alg.get_matrix_and_jobs(pi, tasks, machines_val, n)

                if not ( len(jobs)>= 1):
                    print("jobs is not a list of size larger than 5")          
                    break      

                cmax = makespan(pi, tasks_perturb, machines_val)

                pi = local_search_perturb(pi, cmax,tasks_perturb,machines_val,jobs[:5])

                n_itr +=1
                #print(f"it {n_itr} , cmax {cmax_best}")

                if n_itr % 50 == 0:
                    pi = pi_best
                    cmax = cmax_best

        except Exception as e:
            print("Error:", str(e))  # Print the error message
            cmax_best = 1E10
        #print("finished ",cmax_best)
        
        return cmax_best,n_itr

    
    ############################################### Iterated Local Search ####################################################
    def ils(self,tasks_val, tasks, machines_val):

        cmax_best = 1E10
        #print("run ...")
        try:
            pi, cmax = self.neh(tasks, machines_val, tasks_val) 
            n = len(pi)
            
            pi_best = pi
            cmax_best = cmax
            n_itr = 0
            time_start = time.time()
            while time.time() - time_start < self.time_max and n_itr <self.iter_max:
                #piprim = local_search(pi, tasks_val)
                piprim = local_search(pi, cmax,tasks,machines_val)

                pi = piprim
                cmax = makespan(pi, tasks, machines_val)
                
                if (cmax<cmax_best):
                    pi_best = pi
                    cmax_best = cmax

                pi = perturbation_operator(pi)

                cmax = makespan(pi, tasks, machines_val)

                n_itr +=1
                #print(f"it {n_itr} , cmax {cmax_best}")

                if n_itr % 50 == 0:
                    pi = pi_best
                    cmax = cmax_best

        except Exception as e:
            print("Error:", str(e))  # Print the error message
            cmax_best = 1E10
        #print("finished ",cmax_best)
        
        return cmax_best,n_itr

    ###################################################################### NEH ############################################
    def sum_and_order(self,tasks_val, machines_val, tasks):
        tab = []
        tab1 = []
        for i in range(0, tasks_val):
            tab.append(0)
            tab1.append(0)
        for j in range(0, tasks_val):
            for k in range(0, machines_val):
                tab[j] += tasks[j][k]
        tmp_tab = tab.copy()
        place = 0
        iter = 0
        while(iter != tasks_val):
            max_time = 1
            for i in range(0, tasks_val):
                if(max_time < tab[i]):
                    max_time = tab[i]
                    place = i
            tab[place] = 1
            tab1[iter] = place
            iter = iter + 1
        return tab1


    def insertNEH(self,sequence, position, value):
        new_seq = sequence[:]
        new_seq.insert(position, value)
        return new_seq


    def neh(self,tasks, machines_val, tasks_val):
        order = self.sum_and_order(tasks_val, machines_val, tasks)
        current_seq = [order[0]]
        for i in range(1, tasks_val):
            min_cmax = float("inf")
            for j in range(0, i + 1):
                tmp = self.insertNEH(current_seq, j, order[i])
                cmax_tmp = makespan(tmp, tasks, machines_val)
                if min_cmax > cmax_tmp:
                    best_seq = tmp
                    min_cmax = cmax_tmp
            current_seq = best_seq
        return current_seq, makespan(current_seq, tasks, machines_val)



    def evaluate(self):
        time.sleep(1)
        if self.method == 'gls':
            res = Parallel(n_jobs=self.nc, timeout=self.time_max*1.1)(delayed(self.gls)(x, y, z, a) for x, y, z , a in zip(self.tasks_val, self.tasks, self.machines_val,[None]*len(self.tasks_val)))
            #print(res)
            #print("check")
            print("Average cmax = ",np.mean(res))
        elif self.method == 'ls':
            res = Parallel(n_jobs=self.nc, timeout=self.time_max*1.1)(delayed(self.ls)(x, y, z) for x, y, z in zip(self.tasks_val, self.tasks, self.machines_val))
            #print(res)
            #print("check")
            print("Average cmax = ",np.mean(res)) 
        elif self.method == 'sa':       
            res = Parallel(n_jobs=self.nc, timeout=self.time_max*1.1)(delayed(self.gls)(x, y, z, a) for x, y, z , a in zip(self.tasks_val, self.tasks, self.machines_val,[None]*len(self.tasks_val)))
            #print(res)
            #print("check")
            print("Average cmax = ",np.mean(res))             
        else:
            print(f">> {self.method} has not been implemented !")
        return
    
    def evaluate_test(self, output_file):
        res_all = []
        with open(output_file, 'w') as f:
            print(f"name, n, m, n_iter, t_cost, cmax", file=f)
            for n, m, t, name in zip(self.tasks_val, self.machines_val, self.tasks, self.names):
                time_start = time.time()
                if self.method == 'gls':
                    res, n_iter = self.gls(n, t, m)
                elif self.method == 'ls':
                    res = self.ls(n, t, m)
                    n_iter = 1
                elif self.method == 'sa':       
                    print("none")
                elif self.method == 'ils':       
                    res, n_iter = self.ils(n, t, m)
                else:
                    print(f">> {self.method} has not been implemented !")
                res_all.append(res)
                time_cost = time.time() - time_start
                print(f"{name}, {n}, {m}, {n_iter}, {time_cost:.3f}, {res:.3f}", file=f, flush=True)
                print(f"{name}, {n}, {m}, {n_iter}, {time_cost:.3f}, {res:.3f}")
            print(f"average res = {np.mean(res_all)}", file=f)
        return



def read_ael_instances(path):
    tasks_val_list = [] 
    machines_val_list = [] 
    tasks_list = []
    names = []

    for id in range(1,101):
        filename = path+str(i) + ".txt"
        file = open(filename, "r")

        tasks_val, machines_val = file.readline().split()
        tasks_val = int(tasks_val)
        machines_val = int(machines_val)

        tasks = np.zeros((tasks_val,machines_val))
        for i in range(tasks_val):
            tmp = file.readline().split()
            for j in range(machines_val):
                tasks[i][j] = int(float(tmp[j*2+1]))

        tasks_val_list.append(tasks_val)
        machines_val_list.append(machines_val)
        tasks_list.append(tasks)
        names.append(str(id))

        file.close()

    return tasks_val_list, machines_val_list, tasks_list, names

import os
def read_taillard_instances(path):
    tasks_val_list = [] 
    machines_val_list = [] 
    tasks_list = []
    names = []

    files = os.listdir(path)
    for filename in files:
        file = open(os.path.join(path, filename), "r")

        while True:
            file.readline()
            line = file.readline().strip()
            if line == "":
                break

            tasks_val, machines_val, _, _, _ = map(int, line.split())
            tasks = np.zeros((tasks_val, machines_val))
            file.readline()
            for i in range(machines_val):
                line = file.readline().strip()
                tmp = list(map(int, line.split()))
                for j in range(tasks_val):
                    tasks[j][i] = tmp[j]

            tasks_val_list.append(tasks_val)
            machines_val_list.append(machines_val)
            tasks_list.append(tasks)
            names.append(filename)

        file.close()

    return tasks_val_list, machines_val_list, tasks_list, names

if __name__ == "__main__":
    
    method = 'gls'  
        # ls: local search
        # neh: neh heuristic
        # gls: guided local search
        # sa: simulated annealing
    nc = 4
    iter_max = 1000
    time_max = 60
    ins_type = 'taillard'  # 'random', 'taillard', 'fsp', 'fsp_large'
    ins_path = '../instance/Taillard/'
    resultfile_name = 'results_'+ins_type+'_'+method+'.txt'
    if ins_type == 'random':   
        n,m,tasks,names = read_ael_instances(ins_path)
    elif ins_type == 'taillard':
        n,m,tasks,names = read_taillard_instances(ins_path)

    eva = Evaluation(method,nc,iter_max,time_max,n,m,tasks,names)
    eva.evaluate_test(resultfile_name)
    print(">> testing finished !")
    