"""Evaluate a pheromone update rule designed by EoH against the AS baseline.

Usage
-----
1. Copy the best `update_pheromone` function from EoH results into heuristic.py
   (replacing the template Ant System body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
Both the AS baseline and the EoH heuristic are evaluated so gains are visible.
"""

import importlib
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation


def _load_update_fn(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod.update_pheromone


def _baseline_update(pheromone, ant_tours, tour_costs, best_tour, best_cost,
                     rho, iteration, max_iterations):
    """Ant System reference implementation."""
    n = pheromone.shape[0]
    pheromone = (1.0 - rho) * pheromone
    for tour, cost in zip(ant_tours, tour_costs):
        delta = 1.0 / cost
        for i in range(n):
            u, v = int(tour[i]), int(tour[(i + 1) % n])
            pheromone[u, v] += delta
            pheromone[v, u] += delta
    return pheromone


def _fmt_results(label: str, results: list[dict]) -> list[str]:
    lines = []
    means = []
    for r in results:
        lines.append(
            f"  {label:30s} | instance {r['instance_id']:2d} | "
            f"mean: {r['mean']:8.4f}  std: {r['std']:7.4f}"
        )
        means.append(r['mean'])
    lines.append(
        f"  {'OVERALL ' + label:30s} | mean tour length: {np.mean(means):.4f}"
    )
    return lines


if __name__ == "__main__":
    eva = Evaluation(
        n_cities=50, n_instance=5,
        n_ants=25, iter_max=200,
        alpha=1.0, beta=2.0, rho=0.1,
        n_runs=10,
    )

    print("Evaluating baseline (Ant System) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_baseline_update)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_update = _load_update_fn("heuristic")
    eoh_results = eva.evaluate(eoh_update)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    cfg = "n_cities=50  n_instance=5  n_ants=25  iter_max=200  n_runs=10  rho=0.1"
    header = (f"  {'Heuristic':30s} | {'Instance':11s} | "
              f"{'Mean tour length':>16}  {'Std':>9}")
    sep = "-" * len(header)

    output_lines = [
        "ACO Pheromone Update – Evaluation Results",
        cfg, sep, header, sep,
    ]
    output_lines += _fmt_results("Baseline (Ant System)", baseline_results)
    output_lines += [sep]
    output_lines += _fmt_results("EoH heuristic", eoh_results)
    output_lines += [sep]

    baseline_mean = np.mean([r['mean'] for r in baseline_results])
    eoh_mean = np.mean([r['mean'] for r in eoh_results])
    gap = (eoh_mean - baseline_mean) / baseline_mean * 100

    output_lines.append(
        f"\n  Per-instance improvement (EoH vs Ant System, lower tour length = better):"
    )
    for b, e in zip(baseline_results, eoh_results):
        gain = b['mean'] - e['mean']
        tag = "BETTER" if gain > 1e-6 else ("WORSE " if gain < -1e-6 else "TIE   ")
        pct = gain / b['mean'] * 100
        output_lines.append(
            f"  {tag}  instance {e['instance_id']:2d}  "
            f"gain: {gain:+.4f} ({pct:+.2f}%)  "
            f"({b['mean']:.4f} → {e['mean']:.4f})"
        )
    output_lines.append(
        f"\n  Overall gap vs baseline: {gap:+.2f}%  "
        f"({baseline_mean:.4f} → {eoh_mean:.4f})"
    )

    full_output = "\n".join(output_lines)
    print(full_output)

    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w") as f:
        f.write(full_output + "\n")
    print(f"\nResults saved to {results_path}")
