"""Evaluate a PSO velocity update rule designed by EoH on the full benchmark suite.

Usage
-----
1. Copy the best `update_velocity` function from EoH results into heuristic.py
   (replacing the template standard-PSO body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
The baseline (standard PSO from heuristic_baseline) is evaluated alongside
the EoH heuristic so performance gains are directly visible.
"""

import importlib
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation


def _load_update_velocity(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod.update_velocity


def _baseline_update_velocity(velocities, positions, pbest_positions, pbest_fitness,
                               gbest_position, gbest_fitness, w, c1, c2,
                               bounds, iteration, max_iterations):
    """Standard PSO velocity update reference implementation."""
    pop_size, dim = velocities.shape
    r1 = np.random.rand(pop_size, dim)
    r2 = np.random.rand(pop_size, dim)
    cognitive = c1 * r1 * (pbest_positions - positions)
    social    = c2 * r2 * (gbest_position  - positions)
    return w * velocities + cognitive + social


def _fmt_row(label: str, results: list[dict]) -> list[str]:
    lines = []
    scores = []
    for r in results:
        line = (f"  {label:30s} | {r['name']:12s} {r['dim']:3d}D | "
                f"mean: {r['mean']:12.4f}  std: {r['std']:10.4f}  "
                f"log1p: {r['log1p_mean']:.4f}")
        lines.append(line)
        scores.append(r['log1p_mean'])
    lines.append(f"  {'OVERALL ' + label:30s} | mean log1p score: {np.mean(scores):.4f}")
    return lines


if __name__ == "__main__":
    eva = Evaluation(pop_size=50, max_iterations=500, n_runs=10,
                     w=0.729, c1=1.494, c2=1.494, v_max_ratio=0.2)

    print("Evaluating baseline (standard PSO) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_baseline_update_velocity)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_update_velocity = _load_update_velocity("heuristic")
    eoh_results = eva.evaluate(eoh_update_velocity)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    header = (f"  {'Heuristic':30s} | {'Function':12s} {'Dim':3s} | "
              f"{'Mean best':>14}  {'Std':>12}  {'log1p'}")
    sep = "-" * len(header)

    output_lines = [
        "PSO Velocity Update Rule – Evaluation Results",
        f"pop_size=50  max_iterations=500  n_runs=10  w=0.729  c1=1.494  c2=1.494",
        sep, header, sep,
    ]
    output_lines += _fmt_row("Baseline (standard PSO)", baseline_results)
    output_lines += [sep]
    output_lines += _fmt_row("EoH heuristic", eoh_results)
    output_lines += [sep]

    output_lines.append("\n  Per-benchmark improvement (EoH vs baseline, lower log1p = better):")
    for b, e in zip(baseline_results, eoh_results):
        gain = b['log1p_mean'] - e['log1p_mean']
        tag = "BETTER" if gain > 1e-6 else ("WORSE " if gain < -1e-6 else "TIE   ")
        output_lines.append(
            f"  {tag}  {e['name']:12s} {e['dim']:3d}D  "
            f"gain: {gain:+.4f}  "
            f"({b['mean']:.4f} → {e['mean']:.4f})"
        )

    full_output = "\n".join(output_lines)
    print(full_output)

    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w") as f:
        f.write(full_output + "\n")
    print(f"\nResults saved to {results_path}")
