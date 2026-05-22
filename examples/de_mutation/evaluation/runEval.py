"""Evaluate a DE mutation operator designed by EoH on the full benchmark suite.

Usage
-----
1. Copy the best `mutation` function from EoH results into heuristic.py
   (replacing the template DE/rand/1 body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
The baseline (DE/rand/1 from heuristic_baseline) is evaluated alongside
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


def _load_mutation(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod.mutation


def _baseline_mutation(population, current_idx, best_idx, fitness, F, bounds):
    """DE/rand/1 reference implementation."""
    pop_size, dim = population.shape
    candidates = [i for i in range(pop_size) if i != current_idx]
    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
    return population[r1] + F * (population[r2] - population[r3])


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
    eva = Evaluation(pop_size=50, max_evals=20000, n_runs=10, F=0.5, CR=0.9)

    print("Evaluating baseline (DE/rand/1) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_baseline_mutation)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_mutation = _load_mutation("heuristic")
    eoh_results = eva.evaluate(eoh_mutation)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    header = (f"  {'Heuristic':30s} | {'Function':12s} {'Dim':3s} | "
              f"{'Mean best':>14}  {'Std':>12}  {'log1p'}")
    sep = "-" * len(header)

    output_lines = [
        "DE Mutation Operator – Evaluation Results",
        f"pop_size=50  max_evals=20000  n_runs=10  F=0.5  CR=0.9",
        sep, header, sep,
    ]
    output_lines += _fmt_row("Baseline (DE/rand/1)", baseline_results)
    output_lines += [sep]
    output_lines += _fmt_row("EoH heuristic", eoh_results)
    output_lines += [sep]

    # Compute per-row gain
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
