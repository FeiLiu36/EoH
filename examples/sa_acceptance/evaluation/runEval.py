"""Evaluate an SA acceptance function designed by EoH against the Boltzmann baseline.

Usage
-----
1. Copy the best `acceptance_probability` function from EoH results into heuristic.py
   (replacing the template Boltzmann body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
Both 10-D and 20-D variants of each benchmark are tested.
"""

import importlib
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation


def _load_acceptance_fn(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod.acceptance_probability


def _boltzmann(delta_fitness, temperature, iteration, max_iterations):
    """Classic Boltzmann acceptance criterion — reference implementation."""
    return float(np.exp(-delta_fitness / max(temperature, 1e-10)))


def _fmt_results(label: str, results: list[dict]) -> list[str]:
    lines = []
    scores = []
    for r in results:
        lines.append(
            f"  {label:32s} | {r['name']:12s} {r['dim']:3d}D | "
            f"mean: {r['mean']:12.4f}  std: {r['std']:10.4f}  "
            f"log1p: {r['log1p_mean']:.4f}"
        )
        scores.append(r['log1p_mean'])
    lines.append(
        f"  {'OVERALL ' + label:32s} | mean log1p score: {np.mean(scores):.4f}"
    )
    return lines


if __name__ == "__main__":
    eva = Evaluation(max_iter=5000, sigma_ratio=0.02, T_ratio=1e-3, n_runs=10)

    print("Evaluating baseline (Boltzmann) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_boltzmann)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_fn = _load_acceptance_fn("heuristic")
    eoh_results = eva.evaluate(eoh_fn)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    cfg = "max_iter=5000  sigma_ratio=0.02  T_ratio=1e-3  n_runs=10"
    header = (f"  {'Heuristic':32s} | {'Function':12s} {'Dim':3s} | "
              f"{'Mean best':>14}  {'Std':>12}  {'log1p'}")
    sep = "-" * len(header)

    output_lines = [
        "SA Acceptance Probability - Evaluation Results",
        cfg, sep, header, sep,
    ]
    output_lines += _fmt_results("Baseline (Boltzmann)", baseline_results)
    output_lines += [sep]
    output_lines += _fmt_results("EoH heuristic", eoh_results)
    output_lines += [sep]

    output_lines.append(
        "\n  Per-benchmark improvement (EoH vs Boltzmann, lower log1p = better):"
    )
    for b, e in zip(baseline_results, eoh_results):
        gain = b['log1p_mean'] - e['log1p_mean']
        tag = "BETTER" if gain > 1e-6 else ("WORSE " if gain < -1e-6 else "TIE   ")
        output_lines.append(
            f"  {tag}  {e['name']:12s} {e['dim']:3d}D  "
            f"gain: {gain:+.4f}  "
            f"({b['mean']:.4f} → {e['mean']:.4f})"
        )

    baseline_score = np.mean([r['log1p_mean'] for r in baseline_results])
    eoh_score = np.mean([r['log1p_mean'] for r in eoh_results])
    output_lines.append(
        f"\n  Overall log1p score: baseline {baseline_score:.4f}  →  "
        f"EoH {eoh_score:.4f}  (Δ {eoh_score - baseline_score:+.4f})"
    )

    full_output = "\n".join(output_lines)
    print(full_output)

    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w") as f:
        f.write(full_output + "\n")
    print(f"\nResults saved to {results_path}")
