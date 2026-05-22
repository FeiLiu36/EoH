# Fixed baseline — do not modify.
# Conventional NSGA-II (Deb et al., 2002).
import numpy as np


class Metaheuristic:
    """Conventional NSGA-II baseline.

    Population size N = min(100, budget // 10), rounded down to even.
    Operators: SBX crossover (pc=0.9, eta_c=20) and
    polynomial mutation (pm=1/dim, eta_m=20).
    Selection: binary tournament on (rank, -crowding_distance).
    Survivor selection: non-dominated sort + crowding distance on the
    combined parent+offspring pool of size 2N, keeping the best N.
    Returns the non-dominated solutions in the final population.
    """

    def __init__(self, func, dim, bounds, budget, n_obj):
        self.func   = func
        self.dim    = dim
        self.bounds = bounds
        self.budget = budget
        self.n_obj  = n_obj

    # ------------------------------------------------------------------
    # NSGA-II internals
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_nondominated_sort(F):
        """Return list-of-lists: fronts[0] is Pareto front, etc."""
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set    = [[] for _ in range(n)]
        fronts           = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        return fronts[:-1]  # drop trailing empty list

    @staticmethod
    def _crowding_distance(F, front):
        """Assign crowding distances for indices in `front`."""
        m = len(front)
        dist = np.zeros(m)
        if m <= 2:
            dist[:] = np.inf
            return dist
        F_front = F[front]
        for obj in range(F_front.shape[1]):
            order = np.argsort(F_front[:, obj])
            f_min = F_front[order[0],  obj]
            f_max = F_front[order[-1], obj]
            dist[order[0]]  = np.inf
            dist[order[-1]] = np.inf
            span = f_max - f_min
            if span < 1e-12:
                continue
            for k in range(1, m - 1):
                dist[order[k]] += (F_front[order[k + 1], obj] -
                                   F_front[order[k - 1], obj]) / span
        return dist

    def _tournament(self, rank, crowd, N):
        """Binary tournament; return index of winner."""
        a, b = np.random.randint(N), np.random.randint(N)
        if rank[a] < rank[b]:
            return a
        if rank[b] < rank[a]:
            return b
        return a if crowd[a] >= crowd[b] else b

    def _sbx(self, p1, p2, lo, hi, eta_c=20.0):
        """Simulated binary crossover; returns two children."""
        c1, c2 = p1.copy(), p2.copy()
        for d in range(self.dim):
            if np.random.rand() > 0.5:
                continue
            if abs(p1[d] - p2[d]) < 1e-10:
                continue
            u = np.random.rand()
            beta = ((2 * u) ** (1.0 / (eta_c + 1)) if u <= 0.5
                    else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1)))
            c1[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
            c2[d] = 0.5 * ((1 - beta) * p1[d] + (1 + beta) * p2[d])
        return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

    def _poly_mutation(self, x, lo, hi, eta_m=20.0):
        """Polynomial mutation in-place; returns mutated copy."""
        x = x.copy()
        pm = 1.0 / self.dim
        for d in range(self.dim):
            if np.random.rand() >= pm:
                continue
            u = np.random.rand()
            span = hi[d] - lo[d]
            if span < 1e-12:
                continue
            if u < 0.5:
                delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1.0
            else:
                delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta_m + 1))
            x[d] = np.clip(x[d] + delta * span, lo[d], hi[d])
        return x

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def solve(self):
        lo, hi = self.bounds[0], self.bounds[1]
        N = max(4, (min(100, self.budget // 10) // 2) * 2)  # even, ≥ 4

        # Initialise population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([self.func(x) for x in X])
        evals = N

        while evals < self.budget:
            # --- rank & crowding for current population ---
            fronts = self._fast_nondominated_sort(F)
            rank  = np.empty(N, dtype=int)
            crowd = np.zeros(N)
            for r, front in enumerate(fronts):
                for i in front:
                    rank[i] = r
                cd = self._crowding_distance(F, front)
                for k, i in enumerate(front):
                    crowd[i] = cd[k]

            # --- generate N offspring ---
            offspring_X, offspring_F = [], []
            while len(offspring_X) < N and evals < self.budget:
                p1 = X[self._tournament(rank, crowd, N)]
                p2 = X[self._tournament(rank, crowd, N)]
                c1, c2 = self._sbx(p1, p2, lo, hi)
                for child in (c1, c2):
                    if len(offspring_X) >= N or evals >= self.budget:
                        break
                    child = self._poly_mutation(child, lo, hi)
                    offspring_X.append(child)
                    offspring_F.append(self.func(child))
                    evals += 1

            if not offspring_X:
                break

            # --- survivor selection on combined pool ---
            cX = np.vstack([X, offspring_X])
            cF = np.vstack([F, offspring_F])
            c_fronts = self._fast_nondominated_sort(cF)
            c_rank   = np.empty(len(cX), dtype=int)
            c_crowd  = np.zeros(len(cX))
            for r, front in enumerate(c_fronts):
                for i in front:
                    c_rank[i] = r
                cd = self._crowding_distance(cF, front)
                for k, i in enumerate(front):
                    c_crowd[i] = cd[k]

            selected = []
            for front in c_fronts:
                if len(selected) + len(front) <= N:
                    selected.extend(front)
                else:
                    remaining = N - len(selected)
                    order = sorted(front, key=lambda i: c_crowd[i], reverse=True)
                    selected.extend(order[:remaining])
                if len(selected) >= N:
                    break

            X = cX[selected]
            F = cF[selected]

        # Return non-dominated solutions in final population
        fronts = self._fast_nondominated_sort(F)
        return X[fronts[0]]
