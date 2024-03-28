import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    machine_subset = np.random.choice(m, max(1, int(0.3*m)), replace=False)
    weighted_avg_execution_time = np.average(time_matrix[:, machine_subset], axis=1, weights=np.random.rand(len(machine_subset)))

    perturb_jobs = np.argsort(weighted_avg_execution_time)[-int(0.3*n):]

    new_matrix = time_matrix.copy()
    perturbation_factors = np.random.uniform(0.8, 1.2, size=(len(perturb_jobs), len(machine_subset)))
    new_matrix[perturb_jobs[:, np.newaxis], machine_subset] *= perturbation_factors

    return new_matrix, perturb_jobs