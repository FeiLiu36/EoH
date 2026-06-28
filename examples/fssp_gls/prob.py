# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import time
import sys
import os
import random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


# ── FSSP / GLS engine (pure-Python port of the original numba implementation) ───


def makespan(order, tasks, machines_val):
    times = [0] * machines_val
    for j in order:
        times[0] += tasks[j][0]
        for k in range(1, machines_val):
            if times[k] < times[k - 1]:
                times[k] = times[k - 1]
            times[k] += tasks[j][k]
    return max(times)


def local_search(sequence, cmax_old, tasks, machines_val):
    """One sweep of swap + insert moves (best-accept within the sweep)."""
    new_seq = sequence[:]
    for i in range(len(new_seq)):
        for j in range(i + 1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    for i in range(1, len(new_seq)):
        for j in range(1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    return new_seq


def local_search_perturb(sequence, cmax_old, tasks, machines_val, job):
    """Targeted swap + insert moves restricted to the perturbed jobs."""
    new_seq = sequence[:]
    for i in job:
        for j in range(i + 1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    for i in job:
        for j in range(1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    return new_seq


def sum_and_order(tasks_val, machines_val, tasks):
    """Order jobs by descending total processing time (NEH ordering)."""
    tab = [0] * tasks_val
    tab1 = [0] * tasks_val
    for j in range(tasks_val):
        for k in range(machines_val):
            tab[j] += tasks[j][k]
    place = 0
    it = 0
    while it != tasks_val:
        max_time = 1
        for i in range(tasks_val):
            if max_time < tab[i]:
                max_time = tab[i]
                place = i
        tab[place] = 1
        tab1[it] = place
        it += 1
    return tab1


def neh(tasks, machines_val, tasks_val):
    """NEH constructive heuristic."""
    order = sum_and_order(tasks_val, machines_val, tasks)
    current_seq = [order[0]]
    for i in range(1, tasks_val):
        min_cmax = float("inf")
        best_seq = None
        for j in range(0, i + 1):
            tmp = current_seq[:]
            tmp.insert(j, order[i])
            cmax_tmp = makespan(tmp, tasks, machines_val)
            if min_cmax > cmax_tmp:
                best_seq = tmp
                min_cmax = cmax_tmp
        current_seq = best_seq
    return current_seq, makespan(current_seq, tasks, machines_val)


def gls(tasks_val, tasks, machines_val, time_max, iter_max, heuristic):
    """Guided local search for one flow-shop instance; returns best makespan."""
    cmax_best = 1E10
    random.seed(2024)
    try:
        pi, cmax = neh(tasks, machines_val, tasks_val)
        n = len(pi)

        pi_best = pi
        cmax_best = cmax
        n_itr = 0
        time_start = time.time()
        while time.time() - time_start < time_max and n_itr < iter_max:
            piprim = local_search(pi, cmax, tasks, machines_val)

            pi = piprim
            cmax = makespan(pi, tasks, machines_val)

            if cmax < cmax_best:
                pi_best = pi
                cmax_best = cmax

            tasks_perturb, jobs = heuristic(pi, tasks.copy(), machines_val, n)
            jobs = list(jobs)

            if len(jobs) <= 1:
                return 1E10
            if len(jobs) > 5:
                jobs = jobs[:5]

            cmax = makespan(pi, tasks_perturb, machines_val)

            pi = local_search_perturb(pi, cmax, tasks_perturb, machines_val, jobs)

            n_itr += 1
            if n_itr % 50 == 0:
                pi = pi_best
                cmax = cmax_best

    except Exception:
        cmax_best = 1E10

    return cmax_best


class FSSPGLS(BaseProblem):
    """Flow-Shop Scheduling Problem — Guided Local Search.

    The LLM designs get_matrix_and_jobs, which at each GLS iteration:
      (a) modifies the processing-time matrix to expose bottleneck jobs, and
      (b) selects 2-5 jobs to perturb via targeted local search.

    GLS loop (per iteration):
      1. One sweep of swap + insert local search on the original times.
      2. Call get_matrix_and_jobs to get a modified time matrix and perturb list.
      3. Targeted swap + insert local search on the modified times, restricted
         to the perturb jobs; evaluate makespan on the *original* times.
      4. Every 50 iterations restart from the best-so-far sequence.

    Fitness: average final makespan across all training instances (lower = better).
    """

    template_program = '''
def get_matrix_and_jobs(current_sequence: list, time_matrix: np.ndarray,
                         m: int, n: int) -> tuple:
    """Modify the processing-time matrix and select jobs to perturb.

    Args:
        current_sequence: current permutation of job indices (list of n ints)
        time_matrix:      n*m matrix of processing times (numpy array)
        m:                number of machines
        n:                number of jobs
    Returns:
        new_matrix:   modified n*m processing-time matrix (numpy array)
        perturb_jobs: list of 2-5 job indices to apply targeted local search on
    """
    return time_matrix.copy(), list(range(min(3, n)))
'''

    task_description = (
        "Given a flow-shop scheduling problem with n jobs and m machines, "
        "design a novel guided local search perturbation strategy. "
        "At each iteration the strategy modifies the processing-time matrix "
        "to expose bottleneck jobs and returns a short list of jobs to perturb "
        "via targeted local search. "
        "The goal is to minimise the final makespan."
    )

    def __init__(self, n_inst_eva: int = 3, time_max: float = 30.0,
                 iter_max: int = 1000, timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_inst_eva = n_inst_eva
        self.time_max = time_max
        self.iter_max = iter_max
        self.tasks_val, self.machines_val, self.tasks = GetData(n_inst_eva).load_instances()

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        cmax_list = np.zeros(self.n_inst_eva)
        for i in range(self.n_inst_eva):
            cmax_list[i] = gls(self.tasks_val[i], self.tasks[i],
                               self.machines_val[i], self.time_max,
                               self.iter_max, callable_func)
        return float(np.mean(cmax_list))
