import numpy as np


def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Baseline: penalise jobs that contribute most to the makespan on the critical machine."""
    # Find the bottleneck machine (highest total load)
    machine_loads = time_matrix.sum(axis=0)
    critical_machine = int(np.argmax(machine_loads))

    # Augment processing times on the critical machine
    new_matrix = time_matrix.copy()
    new_matrix[:, critical_machine] *= 1.2

    # Perturb the 3 jobs with the longest processing time on the critical machine
    top_jobs = np.argsort(-time_matrix[:, critical_machine])[:3].tolist()
    return new_matrix, top_jobs
