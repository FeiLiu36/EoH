# import numpy as np


# def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
#     """Baseline: penalise jobs that contribute most to the makespan on the critical machine."""
#     # Find the bottleneck machine (highest total load)
#     machine_loads = time_matrix.sum(axis=0)
#     critical_machine = int(np.argmax(machine_loads))

#     # Augment processing times on the critical machine
#     new_matrix = time_matrix.copy()
#     new_matrix[:, critical_machine] *= 1.2

#     # Perturb the 3 jobs with the longest processing time on the critical machine
#     top_jobs = np.argsort(-time_matrix[:, critical_machine])[:3].tolist()
#     return new_matrix, top_jobs

# EoH heuristic
import numpy as np
def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    machine_subset = np.random.choice(m, max(1, int(0.3*m)), replace=False)
    # randomly select a subset of machines
    weighted_avg_execution_time = np.average(time_matrix[:, machine_subset], axis=1,
    weights=np.random.rand(len(machine_subset)))
    # compute the weighted average execution time
    perturb_jobs = np.argsort(weighted_avg_execution_time)[-int(0.3*n):]
    # sort the last jobs based on the weighted average execution time
    new_matrix = time_matrix.copy()
    perturbation_factors = np.random.uniform(0.8, 1.2, size=(len(perturb_jobs), len(
    machine_subset)))
    # calculate perturbation factors, introduce certain randomness
    new_matrix[perturb_jobs[:, np.newaxis], machine_subset] *= perturbation_factors
    # calculate the final guiding matrix
    return new_matrix, perturb_jobs