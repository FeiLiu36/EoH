import numpy as np
import importlib
import time
from numba import jit
from joblib import Parallel, delayed
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



class Evaluation():
    def __init__(self) -> None:
        self.iter_max = 1000
        self.time_max = 30
        self.tasks_val, self.machines_val, self.tasks = self.read_ael_instances()


    ############################################### Local Search ####################################################
    def ls(self,tasks_val, tasks, machines_val):
        pi0, cmax0 = self.neh(tasks, machines_val, tasks_val) 
        #print("neh results: ",cmax0) 
        pi = pi0
        cmax_old = cmax0
        while True:
            #piprim = local_search(pi, tasks_val)
            piprim = local_search(pi, cmax_old,tasks,machines_val)
            cmax = makespan(piprim, tasks, machines_val)
            if (cmax>=cmax_old):
                break
            else:
                pi = piprim
                cmax_old = cmax
        return pi, cmax_old

    ############################################### Iterated Local Search ####################################################
    def gls(self,tasks_val, tasks, machines_val, alg):
        heuristic_module = importlib.import_module("ael_alg")
        alg = importlib.reload(heuristic_module)
        cmax_best = 1E10
        random.seed(2024)
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

                tasks_perturb, jobs = alg.get_matrix_and_jobs(pi, tasks.copy(), machines_val, n)

                if ( len(jobs) <= 1):
                    print("jobs is not a list of size larger than 1")          
                    return 1E10   
                if  ( len(jobs) > 5):
                    jobs = jobs[:5]

                cmax = makespan(pi, tasks_perturb, machines_val)

                pi = local_search_perturb(pi, cmax,tasks_perturb,machines_val,jobs)

                n_itr +=1
                #print(f"it {n_itr} , cmax {cmax_best}")
                if n_itr % 50 == 0:
                    pi = pi_best
                    cmax = cmax_best

        except Exception as e:
            print("Error:", str(e))  # Print the error message
            cmax_best = 1E10
        #print("finished ",cmax_best)
        
        return cmax_best

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


    def read_ael_instances(self):
        tasks_val_list = [] 
        machines_val_list = [] 
        tasks_list = []

        for i in range(1,65):
            filename = "ael_instances/"+ str(i) + ".txt"
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

            file.close()

        return tasks_val_list, machines_val_list, tasks_list

    def evaluate(self):
        time.sleep(1)
        try:
   

            

            # for task_v, machine, task in zip(tasks_val, machines_val, tasks):
            #     #print(task_v, task, machine)
            #     #best_seq, best_cmax = sim_ann1(tasks_val, tasks, machines_val, 5000, 10)
            #     best_seq, best_cmax =  self.gls(task_v, task, machine, alg)
            #     #best_seq, best_cmax = sim_ann2(tasks_val, tasks, machines_val, T_start, T_end, iter_max)      
            #     #print("normalized best Cmax: ", best_cmax)
            #     best_cmaxs.append(best_cmax)
            #res = np.zeros(len(tasks_val))
            res = Parallel(n_jobs=4, timeout=self.time_max*1.1)(delayed(self.gls)(x, y, z, a) for x, y, z , a in zip(self.tasks_val, self.tasks, self.machines_val,[None]*len(self.tasks_val)))
            #print(res)
            #print("check")
            print("Average cmax = ",np.mean(res))
            return np.mean(res)
        except Exception as e:
            print("Error:",str(e))
            return None



if __name__ == "__main__":
    method = 'ls'
    # alg 
    # ls: local search
    # neh: neh heuristic
    # gls: guided local search
    # sa: simulated annealing

    eva = Evaluation()
    eva.evaluate()
    

