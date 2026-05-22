# Baseline metaheuristic: scalarized (mu+1)-ES with Pareto archiving.
# Replace this class body with the best Metaheuristic class found by EoH.
import numpy as np

class Metaheuristic:

    def __init__(self, func, dim, bounds, budget, n_obj):
        self.func   = func
        self.dim    = dim
        self.bounds = bounds
        self.budget = budget
        self.n_obj  = n_obj

    def solve(self):
        lo, hi = self.bounds[0], self.bounds[1]
        pop_size = min(100, self.budget // 5)
        archive_x, archive_f = [], []

        # Initialize population
        pop = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        pop_f = np.array([self.func(x) for x in pop])
        evals = pop_size
        archive_x.extend(pop.copy())
        archive_f.extend(pop_f.copy())

        while evals < self.budget:
            # Non-dominated sorting and crowding distance
            combined_f = pop_f.copy()
            N = len(pop)
            rank = np.full(N, -1)
            crowding = np.zeros(N)
            current_rank = 0
            remaining = set(range(N))

            while remaining:
                front = []
                for i in remaining:
                    if not any(j in remaining and np.all(combined_f[j] <= combined_f[i]) and np.any(combined_f[j] < combined_f[i]) for j in remaining):
                        front.append(i)
                if not front:
                    front = list(remaining)
                # Assign rank
                for i in front:
                    rank[i] = current_rank
                front_f = combined_f[front]
                # Crowding distance
                if len(front) > 2:
                    for obj in range(self.n_obj):
                        idx = np.argsort(front_f[:, obj])
                        obj_min = front_f[idx[0], obj]
                        obj_max = front_f[idx[-1], obj]
                        if obj_max - obj_min > 1e-10:
                            for k in range(1, len(idx)-1):
                                crowding[front[idx[k]]] += (front_f[idx[k+1], obj] - front_f[idx[k-1], obj]) / (obj_max - obj_min)
                    # Boundary points get infinite crowding
                    crowding[front[idx[0]]] = np.inf
                    crowding[front[idx[-1]]] = np.inf
                else:
                    for i in front:
                        crowding[i] = np.inf
                for i in front:
                    remaining.remove(i)
                current_rank += 1

            # Generate offspring via SBX and polynomial mutation
            offspring_x = []
            for _ in range(pop_size):
                # Tournament selection (binary tournament based on rank then crowding)
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                if rank[idx1] < rank[idx2] or (rank[idx1] == rank[idx2] and crowding[idx1] > crowding[idx2]):
                    parent1 = pop[idx1]
                else:
                    parent1 = pop[idx2]
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                if rank[idx1] < rank[idx2] or (rank[idx1] == rank[idx2] and crowding[idx1] > crowding[idx2]):
                    parent2 = pop[idx1]
                else:
                    parent2 = pop[idx2]

                # Simulated Binary Crossover (SBX)
                child = np.zeros(self.dim)
                for d in range(self.dim):
                    if np.random.rand() < 0.9:
                        if abs(parent1[d] - parent2[d]) > 1e-10:
                            u = np.random.rand()
                            beta = (2*u) ** (1/(20+1)) if u <= 0.5 else (1/(2*(1-u))) ** (1/(20+1))
                            child[d] = 0.5 * ((1+beta)*parent1[d] + (1-beta)*parent2[d])
                        else:
                            child[d] = parent1[d]
                    else:
                        child[d] = parent1[d]
                # Polynomial mutation
                for d in range(self.dim):
                    if np.random.rand() < 1.0/self.dim:
                        u = np.random.rand()
                        delta = (2*u) ** (1/(20+1)) - 1 if u < 0.5 else 1 - (2*(1-u)) ** (1/(20+1))
                        child[d] = child[d] + delta * (hi[d] - lo[d]) / 2
                child = np.clip(child, lo, hi)
                offspring_x.append(child)

            # Evaluate offspring
            offspring_f = []
            for x in offspring_x:
                if evals >= self.budget:
                    break
                f = self.func(x)
                offspring_f.append(f.copy())
                archive_x.append(x.copy())
                archive_f.append(f.copy())
                evals += 1

            if evals >= self.budget:
                break

            # Combine parent and offspring, select best by non-dominated sorting and crowding
            combined_x = list(pop) + list(offspring_x[:len(offspring_f)])
            combined_f = np.array(list(pop_f) + list(offspring_f))
            N_comb = len(combined_x)
            rank_comb = np.full(N_comb, -1)
            crowding_comb = np.zeros(N_comb)
            current_rank = 0
            remaining = set(range(N_comb))

            while remaining and len([i for i in range(N_comb) if rank_comb[i] != -1]) < pop_size:
                front = []
                for i in remaining:
                    if not any(j in remaining and np.all(combined_f[j] <= combined_f[i]) and np.any(combined_f[j] < combined_f[i]) for j in remaining):
                        front.append(i)
                if not front:
                    front = list(remaining)[:pop_size - len([i for i in range(N_comb) if rank_comb[i] != -1])]
                for i in front:
                    rank_comb[i] = current_rank
                front_f = combined_f[front]
                if len(front) > 2:
                    for obj in range(self.n_obj):
                        idx = np.argsort(front_f[:, obj])
                        obj_min = front_f[idx[0], obj]
                        obj_max = front_f[idx[-1], obj]
                        if obj_max - obj_min > 1e-10:
                            for k in range(1, len(idx)-1):
                                crowding_comb[front[idx[k]]] += (front_f[idx[k+1], obj] - front_f[idx[k-1], obj]) / (obj_max - obj_min)
                    crowding_comb[front[idx[0]]] = np.inf
                    crowding_comb[front[idx[-1]]] = np.inf
                else:
                    for i in front:
                        crowding_comb[i] = np.inf
                for i in front:
                    remaining.remove(i)
                current_rank += 1

            # Select pop_size individuals
            selected = []
            for r in range(current_rank):
                front_indices = [i for i in range(N_comb) if rank_comb[i] == r]
                if len(selected) + len(front_indices) <= pop_size:
                    selected.extend(front_indices)
                else:
                    # Sort by crowding distance descending
                    front_indices.sort(key=lambda i: crowding_comb[i], reverse=True)
                    selected.extend(front_indices[:pop_size - len(selected)])
                if len(selected) >= pop_size:
                    break

            pop = np.array([combined_x[i] for i in selected])
            pop_f = np.array([combined_f[i] for i in selected])

        # Final non-dominated filter on archive
        all_f = np.array(archive_f)
        nd = np.ones(len(all_f), dtype=bool)
        for i in range(len(all_f)):
            dominated_by = np.all(all_f <= all_f[i], axis=1) & np.any(all_f < all_f[i], axis=1)
            dominated_by[i] = False
            if dominated_by.any():
                nd[i] = False
        return np.array(archive_x)[nd]
    