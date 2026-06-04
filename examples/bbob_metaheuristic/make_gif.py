"""
GIF showing population evolution in 2D on Rosenbrock using the EoH metaheuristic
from examples/bbob_metaheuristic/evaluation/heuristic.py.

Algorithm: GA (tournament selection + blend crossover) + Simulated Annealing acceptance.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

# ── benchmark (2-D Rosenbrock) ───────────────────────────────────────────────
def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))

FUNC      = rosenbrock
FUNC_NAME = 'Rosenbrock'
DIM       = 2
LO, HI    = -2.048, 2.048   # standard Rosenbrock bounds
BUDGET    = 2000
TRUE_OPT  = np.array([1.0, 1.0])   # global optimum at f=0

# ── instrumented solve (verbatim logic from heuristic.py) ────────────────────
np.random.seed(42)

pop_size = max(20, DIM * 2)          # 10
budget_remaining = BUDGET

pop     = LO + np.random.rand(pop_size, DIM) * (HI - LO)
fitness = np.array([FUNC(ind) for ind in pop])
budget_remaining -= pop_size

best_idx = np.argmin(fitness)
x_best   = pop[best_idx].copy()
f_best   = fitness[best_idx]

T_start      = 100.0
T_end        = 1.0
total_iters  = budget_remaining // pop_size
cooling_rate = (T_start - T_end) / max(total_iters, 1)
T            = T_start

# frames: (pop, fitness, x_best_so_far, f_best_so_far, T, generation)
frames = [(pop.copy(), fitness.copy(), x_best.copy(), f_best, T, 0)]

gen = 0
while budget_remaining > 0 and total_iters > 0:
    new_pop = np.empty_like(pop)
    for i in range(pop_size):
        i1, i2 = np.random.randint(0, pop_size, 2)
        p1 = pop[i1] if fitness[i1] < fitness[i2] else pop[i2]
        i3, i4 = np.random.randint(0, pop_size, 2)
        p2 = pop[i3] if fitness[i3] < fitness[i4] else pop[i4]
        alpha = np.random.rand(DIM)
        child = alpha * p1 + (1 - alpha) * p2
        sigma = (HI - LO) * (T / T_start) * 0.1
        child += np.random.randn(DIM) * sigma
        new_pop[i] = np.clip(child, LO, HI)

    new_fitness = np.array([FUNC(ind) for ind in new_pop])
    budget_remaining -= min(pop_size, budget_remaining)

    for i in range(pop_size):
        delta = new_fitness[i] - fitness[i]
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            pop[i]     = new_pop[i]
            fitness[i] = new_fitness[i]
            if fitness[i] < f_best:
                f_best = fitness[i]
                x_best = pop[i].copy()

    T = max(T - cooling_rate, T_end)
    total_iters -= 1
    gen += 1
    frames.append((pop.copy(), fitness.copy(), x_best.copy(), f_best, T, gen))

# hold the final state a few extra beats
for _ in range(5):
    frames.append(frames[-1])

# thin to keep GIF compact
frames = frames[::2] + frames[-5:]
n_frames = len(frames)

# True global optimum of Rosenbrock: x*=(1,1), f=0 – displayed as a fixed gold star
FINAL_BEST   = TRUE_OPT          # (1.0, 1.0)
FINAL_F_BEST = 0.0
algo_best    = frames[-1][2].copy()
algo_f_best  = frames[-1][3]
print(f"Captured {n_frames} frames")
print(f"True optimum: {FINAL_BEST}  f=0")
print(f"Algorithm best: f(x*) = {algo_f_best:.6f}  at {algo_best}")

# ── pre-compute landscape grid (drawn directly each frame – no coord mismatch) ─
RES = 150
xs = np.linspace(LO, HI, RES)
ys = np.linspace(LO, HI, RES)
XX, YY = np.meshgrid(xs, ys)
ZZ     = 100 * (YY - XX ** 2) ** 2 + (XX - 1) ** 2   # Rosenbrock formula
ZZ_log = np.log1p(ZZ)                                  # log scale for contrast

# ── main figure ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5), dpi=90)
fig.patch.set_facecolor('#f8f9fa')
fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.10)

def draw(fi):
    ax.clear()

    pop_f, fit_f, xb_cur, fb_cur, Tf, gf = frames[fi]

    # landscape drawn in the same axes → coordinates always match
    ax.contourf(XX, YY, ZZ_log, levels=35, cmap='magma_r', zorder=1)
    ax.contour(XX, YY, ZZ_log, levels=14, colors='white',
               linewidths=0.2, alpha=0.25, zorder=2)

    # population dots (color = within-generation fitness rank)
    rank   = np.argsort(np.argsort(fit_f))          # 0 = best
    norm_r = rank / max(rank.max(), 1)
    colors = plt.cm.RdYlGn_r(norm_r)                # green=best, red=worst
    ax.scatter(pop_f[:, 0], pop_f[:, 1],
               s=55, c=colors, edgecolors='white', linewidths=0.5,
               zorder=4, alpha=0.92)

    # current-frame best (white diamond) – shows algorithm progress each frame
    ax.scatter(*xb_cur, s=120, c='white', marker='D',
               edgecolors='#444444', linewidths=0.8, zorder=5,
               label=f'Current best  f={fb_cur:.3f}')

    # fixed gold star at the final best – does NOT move between frames
    ax.scatter(*FINAL_BEST, s=260, c='gold', marker='*',
               edgecolors='black', linewidths=0.8, zorder=6,
               label=f'True optimum (1,1)  f=0')

    # temperature bar (axes-coordinate strip below the plot)
    t_frac = (Tf - T_end) / (T_start - T_end)
    ax.add_patch(plt.Rectangle((0, -0.07), 1, 0.035,
                                transform=ax.transAxes, fc='#333333',
                                ec='none', clip_on=False, zorder=5))
    ax.add_patch(plt.Rectangle((0, -0.07), t_frac, 0.035,
                                transform=ax.transAxes,
                                fc=plt.cm.coolwarm(t_frac),
                                ec='none', clip_on=False, zorder=6))
    ax.text(0.5, -0.052, f'Temperature  T = {Tf:.1f}',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=7.5, color='white', fontweight='bold',
            clip_on=False, zorder=7)

    # legend
    leg = [
        mpatches.Patch(color=plt.cm.RdYlGn_r(0.0), label='Best in pop'),
        mpatches.Patch(color=plt.cm.RdYlGn_r(1.0), label='Worst in pop'),
        plt.Line2D([0], [0], marker='D', color='none', markerfacecolor='white',
                   markeredgecolor='#444', markersize=8,
                   label=f'Current best  f={fb_cur:.3f}'),
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='gold',
                   markeredgecolor='black', markersize=13,
                   label=f'True optimum (1,1)  f=0'),
    ]
    ax.legend(handles=leg, loc='upper left', fontsize=7,
              framealpha=0.75, facecolor='#111111', edgecolor='#888888',
              labelcolor='white')

    # evals derived from generation counter stored in the frame
    evals = pop_size + gf * pop_size
    ax.set_title(
        f'{FUNC_NAME} 2D — EoH Metaheuristic (GA + SA)\n'
        f'Gen {gf}  |  Evals {min(evals, BUDGET)}  |  Current best f(x*) = {fb_cur:.4f}',
        fontsize=9.5, fontweight='bold', color='#1a1a2e', pad=5
    )
    ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
    ax.set_xlabel('x₁', fontsize=10); ax.set_ylabel('x₂', fontsize=10)
    ax.tick_params(labelsize=7.5)

# ── save ──────────────────────────────────────────────────────────────────────
ani = FuncAnimation(fig, draw, frames=n_frames, interval=220,
                    repeat=True, repeat_delay=2000)
out = 'bbob_metaheuristic.gif'
ani.save(out, writer=PillowWriter(fps=4))
plt.close()
print(f"Saved: {out}  ({n_frames} frames)")
