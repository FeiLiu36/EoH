# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


# ── FSSP primitives ────────────────────────────────────────────────────────────

def _makespan(order, tasks):
    """Compute makespan for a job permutation on a flow-shop."""
    n_machines = tasks.shape[1]
    c = np.zeros(n_machines)
    for job in order:
        c[0] += tasks[job, 0]
        for k in range(1, n_machines):
            if c[k] < c[k - 1]:
                c[k] = c[k - 1]
            c[k] += tasks[job, k]
    return float(c[-1])


def _neh(tasks):
    """NEH constructive heuristic — good initial solution for FSSP."""
    n_jobs = len(tasks)
    order = np.argsort(-tasks.sum(axis=1)).tolist()
    seq = [order[0]]
    for i in range(1, n_jobs):
        best_pos, best_c = 0, float('inf')
        for j in range(i + 1):
            s = seq[:j] + [order[i]] + seq[j:]
            c = _makespan(s, tasks)
            if c < best_c:
                best_c = c
                best_pos = j
        seq.insert(best_pos, order[i])
    return seq, _makespan(seq, tasks)


def _local_search(seq, tasks, t_end=None):
    """Swap + insert local search until no improving move (with optional time cap)."""
    best = seq[:]
    best_c = _makespan(best, tasks)
    improved = True
    while improved:
        if t_end and time.time() > t_end:
            break
        improved = False
        n = len(best)
        for i in range(n):
            for j in range(i + 1, n):
                s = best[:]
                s[i], s[j] = s[j], s[i]
                c = _makespan(s, tasks)
                if c < best_c - 1e-10:
                    best, best_c, improved = s, c, True
        for i in range(n):
            job = best[i]
            for j in range(n):
                if i == j:
                    continue
                s = best[:]
                s.pop(i)
                s.insert(j, job)
                c = _makespan(s, tasks)
                if c < best_c - 1e-10:
                    best, best_c, improved = s, c, True
    return best, best_c


def _local_search_targeted(seq, tasks, jobs):
    """Insert-only local search restricted to the given job indices."""
    best = seq[:]
    best_c = _makespan(best, tasks)
    n = len(best)
    for job in jobs:
        if job not in best:
            continue
        i = best.index(job)
        for j in range(n):
            if i == j:
                continue
            s = best[:]
            s.pop(i)
            s.insert(j, job)
            c = _makespan(s, tasks)
            if c < best_c - 1e-10:
                best, best_c = s, c
    return best, best_c


class FSSPGLS(BaseProblem):
    """Flow-Shop Scheduling Problem — Guided Local Search.

    The LLM designs get_matrix_and_jobs, which at each GLS iteration:
      (a) modifies the processing-time matrix to expose bottleneck jobs, and
      (b) selects 2–5 jobs to perturb via targeted local search.

    GLS loop (per iteration):
      1. Full swap+insert local search on the original times.
      2. Call get_matrix_and_jobs to get a modified time matrix and perturb list.
      3. Targeted insert-LS on the modified times, restricted to perturb jobs.
      4. Record makespan on the *original* times.

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

    def __init__(self, n_jobs: int = 20, n_machines: int = 5,
                 n_instance: int = 3, iter_max: int = 100, time_max: float = 10.0,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n_instance = n_instance
        self.iter_max = iter_max
        self.time_max = time_max
        self.instance_data = GetData(n_instance, n_jobs, n_machines).generate_instances()

    def _gls(self, tasks, heuristic):
        seq, cmax = _neh(tasks)
        best_seq, best_cmax = seq[:], cmax

        t_end = time.time() + self.time_max
        for _ in range(self.iter_max):
            if time.time() > t_end:
                break

            # Full LS on original times
            seq, cmax = _local_search(seq, tasks, t_end)
            if cmax < best_cmax:
                best_seq, best_cmax = seq[:], cmax

            # Perturbation via LLM-designed heuristic
            result = heuristic(seq[:], tasks.copy(), self.n_machines, self.n_jobs)
            new_matrix, perturb_jobs = result
            new_matrix = np.asarray(new_matrix, dtype=float)

            # Validate outputs
            if new_matrix.shape != tasks.shape:
                continue
            perturb_jobs = list(perturb_jobs)
            if len(perturb_jobs) < 2:
                continue
            perturb_jobs = [int(j) for j in perturb_jobs[:5]
                            if 0 <= int(j) < self.n_jobs]
            if len(perturb_jobs) < 2:
                continue

            # Targeted LS on modified times
            seq, _ = _local_search_targeted(seq, new_matrix, perturb_jobs)
            cmax = _makespan(seq, tasks)          # evaluate on original
            if cmax < best_cmax:
                best_seq, best_cmax = seq[:], cmax

        return best_cmax

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        makespans = []
        for tasks in self.instance_data:
            makespans.append(self._gls(tasks, callable_func))
        return float(np.mean(makespans))
