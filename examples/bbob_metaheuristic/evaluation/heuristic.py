# Baseline metaheuristic: (1+lambda)-ES with 1/5-rule step-size adaptation.
# Replace this class body with the best Metaheuristic class found by EoH.
import numpy as np

class Metaheuristic:

    def __init__(self, func, dim, bounds, budget):
        self.func   = func
        self.dim    = dim
        self.bounds = bounds
        self.budget = budget

    def solve(self):
        np.random.seed()
        lower = self.bounds[0]
        upper = self.bounds[1]
        budget_remaining = self.budget
        
        # Population size
        pop_size = max(10, self.dim * 2)
        # Initialize population
        pop = lower + np.random.rand(pop_size, self.dim) * (upper - lower)
        fitness = np.array([self.func(ind) for ind in pop])
        budget_remaining -= pop_size
        
        best_idx = np.argmin(fitness)
        x_best = pop[best_idx].copy()
        f_best = fitness[best_idx]
        
        # Temperature schedule
        T_start = 100.0
        T_end = 1.0
        total_iterations = budget_remaining // pop_size
        if total_iterations == 0:
            total_iterations = 1
        cooling_rate = (T_start - T_end) / total_iterations
        T = T_start
        
        while budget_remaining > 0 and total_iterations > 0:
            # Crossover and mutation to generate new population
            new_pop = np.empty_like(pop)
            for i in range(pop_size):
                # Select two parents via tournament
                idx1 = np.random.randint(0, pop_size)
                idx2 = np.random.randint(0, pop_size)
                if fitness[idx1] < fitness[idx2]:
                    p1 = pop[idx1]
                else:
                    p1 = pop[idx2]
                idx3 = np.random.randint(0, pop_size)
                idx4 = np.random.randint(0, pop_size)
                if fitness[idx3] < fitness[idx4]:
                    p2 = pop[idx3]
                else:
                    p2 = pop[idx4]
                # Crossover: blend
                alpha = np.random.rand(self.dim)
                child = alpha * p1 + (1 - alpha) * p2
                # Mutation: Gaussian with scale proportional to temperature
                sigma = (upper - lower) * (T / T_start) * 0.1
                child += np.random.randn(self.dim) * sigma
                child = np.clip(child, lower, upper)
                new_pop[i] = child
            
            # Evaluate new population
            new_fitness = np.array([self.func(ind) for ind in new_pop])
            evaluations_used = min(pop_size, budget_remaining)
            budget_remaining -= evaluations_used
            
            # Simulated annealing acceptance
            for i in range(pop_size):
                delta = new_fitness[i] - fitness[i]
                if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    pop[i] = new_pop[i]
                    fitness[i] = new_fitness[i]
                    if fitness[i] < f_best:
                        f_best = fitness[i]
                        x_best = pop[i].copy()
            
            # Cool temperature
            T = max(T - cooling_rate, T_end)
            total_iterations -= 1
        
        return x_best
    