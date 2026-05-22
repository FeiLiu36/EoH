"""Evaluate a Tabu Search move-scoring function designed by EoH against the baseline.

Usage
-----
1. Copy the best `score_moves` function from EoH results into heuristic.py
   (replacing the template body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
Both 20-node and 30-node random Euclidean TSP instances are tested.
"""

import importlib
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation


def _load_score_fn(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod.score_moves


def _baseline(delta_costs, is_tabu_mask, best_cost, current_cost,
              tabu_ages, iteration, max_iterations):
    """Classic best non-tabu move with simple aspiration criterion."""
    scores = np.full(len(delta_costs), -np.inf)
    non_tabu = ~is_tabu_mask
    scores[non_tabu] = -delta_costs[non_tabu]
    aspiration = is_tabu_mask & (current_cost + delta_costs < best_cost)
    scores[aspiration] = -delta_costs[aspiration] + 1e6
    return scores


def _fmt_results(label: str, results: list[dict]) -> list[str]:
    lines = []
    per_node_scores = []
    for r in results:
        lines.append(
            f"  {label:36s} | {r['n_nodes']:2d}-node | "
            f"mean: {r['mean']:10.3f}  std: {r['std']:8.3f}  "
            f"per-node: {r['mean_per_node']:.3f}"
        )
        per_node_scores.append(r['mean_per_node'])
    lines.append(
        f"  {'OVERALL ' + label:36s} | mean per-node cost: {np.mean(per_node_scores):.3f}"
    )
    return lines


if __name__ == "__main__":
    eva = Evaluation(n_iter=500, tabu_tenure=7, n_runs=10)

    print("Evaluating baseline (best non-tabu + aspiration) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_baseline)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_fn = _load_score_fn("heuristic")
    eoh_results = eva.evaluate(eoh_fn)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    cfg = "n_iter=500  tabu_tenure=7  n_runs=10  instances: 10×20-node + 10×30-node"
    header = (
        f"  {'Heuristic':36s} | {'Size':7s} | "
        f"{'Mean cost':>12}  {'Std':>10}  {'Per-node':>10}"
    )
    sep = "-" * len(header)

    output_lines = [
        "Tabu Search TSP — Move Scoring Function Evaluation",
        cfg, sep, header, sep,
    ]
    output_lines += _fmt_results("Baseline (best non-tabu)", baseline_results)
    output_lines += [sep]
    output_lines += _fmt_results("EoH heuristic", eoh_results)
    output_lines += [sep]

    output_lines.append(
        "\n  Per-group improvement (EoH vs Baseline, lower per-node cost = better):"
    )
    for b, e in zip(baseline_results, eoh_results):
        gain = b['mean_per_node'] - e['mean_per_node']
        tag = "BETTER" if gain > 1e-6 else ("WORSE " if gain < -1e-6 else "TIE   ")
        output_lines.append(
            f"  {tag}  {e['n_nodes']:2d}-node  "
            f"gain: {gain:+.3f}  "
            f"({b['mean_per_node']:.3f} → {e['mean_per_node']:.3f})"
        )

    baseline_score = np.mean([r['mean_per_node'] for r in baseline_results])
    eoh_score = np.mean([r['mean_per_node'] for r in eoh_results])
    output_lines.append(
        f"\n  Overall mean per-node cost: baseline {baseline_score:.3f}  →  "
        f"EoH {eoh_score:.3f}  (Δ {eoh_score - baseline_score:+.3f})"
    )

    full_output = "\n".join(output_lines)
    print(full_output)

    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w") as f:
        f.write(full_output + "\n")
    print(f"\nResults saved to {results_path}")
