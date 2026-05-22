"""Evaluate a DE crossover operator designed by EoH on the full benchmark suite.

Usage
-----
1. Copy the best `crossover` function from EoH results into heuristic.py
   (replacing the template binomial crossover body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
The baseline (binomial crossover) is evaluated alongside the EoH heuristic
so performance gains are directly visible across 50-D, 100-D, and 200-D.
"""

import importlib
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation


def _load_crossover(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod.crossover


def _baseline_crossover(target, mutant, CR, generation, max_generations,
                        fitness_target, fitness_best):
    """Standard binomial crossover reference implementation."""
    dim = len(target)
    mask = np.random.rand(dim) < CR
    mask[np.random.randint(dim)] = True
    return np.where(mask, mutant, target)


def _fmt_rows(label: str, results: list[dict]) -> list[str]:
    lines = []
    scores = []
    for r in results:
        line = (f"  {label:32s} | {r['name']:12s} {r['dim']:3d}D | "
                f"mean: {r['mean']:14.4f}  std: {r['std']:12.4f}  "
                f"log1p: {r['log1p_mean']:.4f}")
        lines.append(line)
        scores.append(r['log1p_mean'])
    lines.append(f"  {'OVERALL ' + label:32s} | mean log1p score: {np.mean(scores):.4f}")
    return lines


if __name__ == "__main__":
    eva = Evaluation(pop_size=50, max_evals=100000, n_runs=20, F=0.8, CR=0.9)

    print("Evaluating baseline (binomial crossover) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_baseline_crossover)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_crossover = _load_crossover("heuristic")
    eoh_results = eva.evaluate(eoh_crossover)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    header = (f"  {'Heuristic':32s} | {'Function':12s} {'Dim':3s} | "
              f"{'Mean best':>16}  {'Std':>14}  {'log1p'}")
    sep = "-" * len(header)

    output_lines = [
        "DE Crossover Operator (100-D) – Evaluation Results",
        f"pop_size=50  max_evals=100000  n_runs=20  F=0.8  CR=0.9",
        sep, header, sep,
    ]
    output_lines += _fmt_rows("Baseline (binomial)", baseline_results)
    output_lines += [sep]
    output_lines += _fmt_rows("EoH heuristic", eoh_results)
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
