"""
Generate a step-by-step GIF of TSP route construction using the EoH-designed
score-based heuristic from examples/tsp_construct/evaluation/heuristic.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

# ── instance ────────────────────────────────────────────────────────────────
np.random.seed(2024)
N = 15
START_NODE = 11
coords = np.random.rand(N, 2)
distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)


# ── helpers ─────────────────────────────────────────────────────────────────
def generate_neighborhood_matrix(c):
    n = len(c)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        mat[i] = np.argsort(np.linalg.norm(c[i] - c, axis=1))
    return mat

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    threshold = 0.7
    scores = {}
    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance_to_unvisited = np.mean(all_distances) if len(all_distances) > 0 else 0.0
        std_dev_distance_to_unvisited = np.std(all_distances) if len(all_distances) > 0 else 0.0
        score = (0.4 * distance_matrix[current_node][node]
                 - 0.3 * average_distance_to_unvisited
                 + 0.2 * std_dev_distance_to_unvisited
                 - 0.1 * distance_matrix[destination_node][node])
        scores[node] = score
    if min(scores.values()) > threshold:
        next_node = min(unvisited_nodes, key=lambda node: distance_matrix[current_node][node])
    else:
        next_node = min(scores, key=scores.get)
    return next_node


def partial_cost(route_list):
    c = 0.0
    for j in range(len(route_list) - 1):
        c += distances[route_list[j], route_list[j + 1]]
    return c


def full_tour_cost(route_arr):
    c = partial_cost(list(route_arr))
    c += distances[route_arr[-1], route_arr[0]]
    return c


# ── run heuristic, record one state per step ─────────────────────────────────
neighbor_matrix = generate_neighborhood_matrix(coords)

route = np.zeros(N, dtype=int)
route[0] = START_NODE          # fix: initialise route[0] to the actual start node
current_node = START_NODE
destination_node = START_NODE

# Each state: dict with keys  route_so_far, current, next_chosen, closed
states = []
states.append(dict(route_so_far=[START_NODE], current=START_NODE, next_chosen=None, closed=False))

for i in range(1, N - 1):
    near = neighbor_matrix[current_node][1:]
    mask = ~np.isin(near, route[:i])   # route[0] is now START_NODE, not 0
    unvisited = near[mask]
    unvisited = unvisited[: min(N, unvisited.size)]

    nxt = select_next_node(current_node, destination_node, unvisited, distances)

    # "about-to-move" frame: show chosen next node before advancing
    states.append(dict(route_so_far=route[:i].tolist(), current=current_node,
                       next_chosen=nxt, closed=False))

    current_node = nxt
    route[i] = current_node

    # "just-moved" frame
    states.append(dict(route_so_far=route[:i + 1].tolist(), current=current_node,
                       next_chosen=None, closed=False))

# last unvisited node
mask = ~np.isin(np.arange(N), route[:N - 1])
last_node = int(np.arange(N)[mask][0])
route[N - 1] = last_node

states.append(dict(route_so_far=route[:N - 1].tolist(), current=route[N - 2],
                   next_chosen=last_node, closed=False))
states.append(dict(route_so_far=route.tolist(), current=last_node,
                   next_chosen=None, closed=False))

# closing edge back to start
states.append(dict(route_so_far=route.tolist(), current=last_node,
                   next_chosen=START_NODE, closed=False))
states.append(dict(route_so_far=route.tolist(), current=START_NODE,
                   next_chosen=None, closed=True))

# hold final frame a bit longer by duplicating it
for _ in range(4):
    states.append(states[-1])

total_cost = full_tour_cost(route)


# ── drawing ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#f8f9fa')


def draw(frame_idx):
    ax.clear()
    ax.set_facecolor('#f8f9fa')
    s = states[frame_idx]
    r = s['route_so_far']
    cur = s['current']
    nxt = s['next_chosen']
    closed = s['closed']

    visited = set(r)
    unvisited = [i for i in range(N) if i not in visited]

    # ── draw partial route edges ──
    if len(r) > 1:
        rx = [coords[v, 0] for v in r]
        ry = [coords[v, 1] for v in r]
        ax.plot(rx, ry, '-', color='#2b7bba', linewidth=2, alpha=0.85, zorder=2)

    # ── closing edge when tour is complete ──
    if closed:
        ax.plot([coords[r[-1], 0], coords[r[0], 0]],
                [coords[r[-1], 1], coords[r[0], 1]],
                '-', color='#2b7bba', linewidth=2, alpha=0.85, zorder=2)

    # ── dashed arrow to next chosen node ──
    if nxt is not None:
        ax.annotate(
            '', xy=coords[nxt], xytext=coords[cur],
            arrowprops=dict(arrowstyle='->', color='#e04040',
                            lw=2, linestyle='dashed',
                            connectionstyle='arc3,rad=0.08'),
            zorder=7
        )

    # ── unvisited nodes ──
    if unvisited:
        ax.scatter(coords[unvisited, 0], coords[unvisited, 1],
                   s=90, c='#cccccc', edgecolors='#888888', linewidths=1, zorder=3)
        for u in unvisited:
            ax.text(coords[u, 0], coords[u, 1] + 0.028, str(u),
                    ha='center', va='bottom', fontsize=8, color='#555555')

    # ── visited (non-current) nodes ──
    prev_vis = [v for v in visited if v != cur and v != START_NODE]
    if prev_vis:
        ax.scatter(coords[prev_vis, 0], coords[prev_vis, 1],
                   s=110, c='#5b9bd5', edgecolors='#1a4f82', linewidths=1.2, zorder=4)
        for v in prev_vis:
            ax.text(coords[v, 0], coords[v, 1] + 0.028, str(v),
                    ha='center', va='bottom', fontsize=8, color='#1a4f82')

    # ── start node ──
    ax.scatter(*coords[START_NODE], s=220, c='#27ae60', edgecolors='#145a32',
               linewidths=1.5, zorder=6, marker='*')
    ax.text(coords[START_NODE, 0], coords[START_NODE, 1] + 0.028, str(START_NODE),
            ha='center', va='bottom', fontsize=9, color='#145a32', fontweight='bold')

    # ── current node (if not start) ──
    if cur != START_NODE:
        ax.scatter(*coords[cur], s=160, c='#e74c3c', edgecolors='#922b21',
                   linewidths=1.5, zorder=6)
        ax.text(coords[cur, 0], coords[cur, 1] + 0.028, str(cur),
                ha='center', va='bottom', fontsize=9, color='#922b21', fontweight='bold')

    # ── next chosen highlight ──
    if nxt is not None and nxt != cur:
        ax.scatter(*coords[nxt], s=160, c='#f39c12', edgecolors='#9a6007',
                   linewidths=1.5, zorder=5)

    # ── title / cost info ──
    step_num = len(r) - 1
    if closed:
        ax.set_title(
            f'TSP Construction — Complete\n'
            f'Tour cost: {total_cost:.4f}  ({N} cities, EoH score-based heuristic)',
            fontsize=11, fontweight='bold', color='#1a1a2e'
        )
    elif nxt is not None:
        ax.set_title(
            f'TSP Construction — EoH Score-Based Heuristic\n'
            f'Step {step_num}: selecting next node → {nxt}  '
            f'(partial cost: {partial_cost(r):.4f})',
            fontsize=11, fontweight='bold', color='#1a1a2e'
        )
    else:
        ax.set_title(
            f'TSP Construction — EoH Score-Based Heuristic\n'
            f'Step {step_num}: at node {cur}  '
            f'(partial cost: {partial_cost(r):.4f})',
            fontsize=11, fontweight='bold', color='#1a1a2e'
        )

    # ── legend ──
    legend_handles = [
        mpatches.Patch(color='#27ae60', label=f'Start node ({START_NODE})'),
        mpatches.Patch(color='#e74c3c', label='Current node'),
        mpatches.Patch(color='#f39c12', label='Next selected node'),
        mpatches.Patch(color='#5b9bd5', label='Visited nodes'),
        mpatches.Patch(color='#cccccc', label='Unvisited nodes'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8,
              framealpha=0.85, edgecolor='#aaaaaa')

    ax.set_xlim(-0.07, 1.07)
    ax.set_ylim(-0.07, 1.12)
    ax.set_xlabel('X coordinate', fontsize=10)
    ax.set_ylabel('Y coordinate', fontsize=10)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(labelsize=8)
    fig.tight_layout()


ani = FuncAnimation(fig, draw, frames=len(states), interval=600, repeat=True, repeat_delay=2500)

out_path = 'tsp_construct.gif'
ani.save(out_path, writer=PillowWriter(fps=1.6))
plt.close()
print(f"Saved: {out_path}  ({len(states)} frames, {N} cities)")
print(f"Final tour cost: {total_cost:.4f}")
print(f"Tour: {route.tolist()}")
