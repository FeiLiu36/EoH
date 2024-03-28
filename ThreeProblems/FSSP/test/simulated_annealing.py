import numpy as np
import random
import math
from numba import jit
import time

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


def insert(sequence, tasks_val):
    a = random.randrange(0, tasks_val)
    b = random.randrange(0, tasks_val)
    new_seq = sequence[:]
    new_seq.remove(a)
    new_seq.insert(b, a)
    return new_seq


def swap(list, tasks_val):
    a = random.randrange(0, tasks_val)
    b = random.randrange(0, tasks_val)
    tmp_list = list.copy()
    tmp = tmp_list[a]
    tmp_list[a] = tmp_list[b]
    tmp_list[b] = tmp
    return tmp_list


def probability(Cold, Cnew, Temp):
    if Cnew < Cold:
        prob = 1
    else:
        prob = math.exp((Cold-Cnew)/Temp)
    return prob

def annealing1(T, u):
    return T*u

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

############################################### CLASSICAL APPROACH ####################################################
def sim_ann1(tasks_val, tasks, machines_val, T_start, T_end):
    pi0, cmax0 = neh(tasks, machines_val, tasks_val)  
    pi = pi0
    cmax_old = cmax0
    T0 = T_start
    T = T0
    u = 0.99
    Tgr = T_end
    while (T >= Tgr):
        #piprim = local_search(pi, tasks_val)
        piprim = local_search(pi, cmax_old,tasks,machines_val)
        cmax = makespan(piprim, tasks, machines_val)
        p = probability(cmax_old, cmax, T)
        s = random.random()
        if p >= s:
            pi = piprim
            cmax_old = cmax
            T = annealing1(T, u)
        else:
            T = annealing1(T, u)
    return pi, cmax_old


############################################# DIFFERENT ANNEALING FUNCTION ###########################################
def annealing2(T, k, kmax):
    return T*(k/kmax)

def sim_ann2(tasks_val, tasks, machines_val, T_start, T_end, iter_val):
    pi0, cmax0 = neh(tasks, machines_val, tasks_val)
    pi = pi0
    cmax_old = cmax0
    T0 = T_start
    T = T0
    u = 0.99
    Tgr = T_end
    iter = 0
    max_iter = iter_val

    for i in range(0, max_iter):
        iter = iter + 1
        piprim = local_search(pi, cmax_old)
        cmax = makespan(piprim, tasks, machines_val)
        p = probability(cmax_old, cmax, T)
        s = random.random()
        if p >= s:
            pi = piprim
            cmax_old = cmax
            T = annealing2(T, iter, max_iter)
        else:
            T = annealing2(T, iter, max_iter)
        if T == 0:
            break
    return pi, cmax_old



############################################### Local Search ####################################################
def ls(tasks_val, tasks, machines_val):
    pi0, cmax0 = neh(tasks, machines_val, tasks_val) 
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
def gls(tasks_val, tasks, machines_val):
    pi0, cmax0 = neh(tasks, machines_val, tasks_val) 
    print("neh results: ",cmax0) 
    pi = pi0
    cmax_old = cmax0
    while True:
        #piprim = local_search(pi, tasks_val)
        piprim = local_search(pi, cmax_old)
        cmax = makespan(piprim, tasks, machines_val)
        if (cmax>=cmax_old):
            break
        else:
            pi = piprim
            cmax_old = cmax
    return pi, cmax_old

###################################################################### NEH ############################################
def sum_and_order(tasks_val, machines_val, tasks):
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


def insertNEH(sequence, position, value):
    new_seq = sequence[:]
    new_seq.insert(position, value)
    return new_seq


def neh(tasks, machines_val, tasks_val):
    order = sum_and_order(tasks_val, machines_val, tasks)
    current_seq = [order[0]]
    for i in range(1, tasks_val):
        min_cmax = float("inf")
        for j in range(0, i + 1):
            tmp = insertNEH(current_seq, j, order[i])
            cmax_tmp = makespan(tmp, tasks, machines_val)
            if min_cmax > cmax_tmp:
                best_seq = tmp
                min_cmax = cmax_tmp
        current_seq = best_seq
    return current_seq, makespan(current_seq, tasks, machines_val)


def read_data(filename):
    file = open(filename, "r")

    tasks_val, machines_val = file.readline().split()
    tasks_val = int(tasks_val)
    machines_val = int(machines_val)

    tasks = np.zeros((tasks_val,machines_val))
    for i in range(tasks_val):
        tmp = file.readline().split()
        for j in range(machines_val):
            tasks[i][j] = int(tmp[j*2+1])

    print("Number of tasks: ", tasks_val)
    print("Number of machines: ", machines_val)
    print("Tasks: \n", tasks)
    file.close()
    return tasks_val, machines_val, tasks

def read_ael_instances():
    tasks_val_list = [] 
    machines_val_list = [] 
    tasks_list = []

    for i in range(1,101):
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


if __name__ == "__main__":
    #tasks_val, machines_val, tasks = read_data("data/small/VFR40_5_5_Gap.txt")


    ##########
    # 1 - Annealing functions comparision
    # 2 - Number of operations comparision
    # 3 - Influence rejection the probability 1 on results
    # 4 - Influence rejection the probability when Cmax = CmaxOld
    # 5 - Influence Tstart and Tstop on results
    MODE = 1  # which mode should be executed
    ##########

    # T_start = 5000
    # T_end = 10
    iter_max = 1000
 
    best_cmaxs = []
    tasks_val, machines_val, tasks = read_ael_instances()

    for task_v, machine, task in zip(tasks_val, machines_val, tasks):
        #print(task_v, task, machine)
        #best_seq, best_cmax = sim_ann1(tasks_val, tasks, machines_val, 5000, 10)
        best_seq, best_cmax =  ls(task_v, task, machine)
        #best_seq, best_cmax = sim_ann2(tasks_val, tasks, machines_val, T_start, T_end, iter_max)      
        #print("normalized best Cmax: ", best_cmax)
        best_cmaxs.append(best_cmax)
    print("Average cmax = ",np.average(best_cmaxs))

