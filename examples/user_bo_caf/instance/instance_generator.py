from botorch.test_functions import Hartmann, Ackley, Cosine8, Rastrigin, \
Rosenbrock, Levy, Powell, Shekel, ThreeHumpCamel, StyblinskiTang, Griewank
import torch
import pickle

device = torch.device("cpu")
dtype = torch.double

real_problem_list = [
Ackley(dim=2, negate=True).to(device=device, dtype=dtype),
Rastrigin(dim=2, negate=True).to(device=device, dtype=dtype),
# Griewank(negate=True).to(device=device, dtype=dtype),
#Rosenbrock(dim=2, negate=True).to(device=device, dtype=dtype),
# Levy(negate=True).to(device=device, dtype=dtype),
# ThreeHumpCamel(negate=True).to(device=device, dtype=dtype),
# StyblinskiTang(negate=True).to(device=device, dtype=dtype),
# Hartmann(dim=3, negate=True).to(device=device, dtype=dtype),
# Powell(negate=True).to(device=device, dtype=dtype),
# Shekel(negate=True).to(device=device, dtype=dtype),
# Hartmann(dim=6, negate=True).to(device=device, dtype=dtype),
# Cosine8(negate=False).to(device=device, dtype=dtype),
]

# Save the real_problem_list to a pickle file
with open("instance/botorch_problem.pkl", "wb") as f:
    pickle.dump(real_problem_list, f)