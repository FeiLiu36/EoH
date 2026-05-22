"""Evaluate a parent selection operator designed by EoH on the full benchmark suite.

Usage
-----
1. Copy the best `select` function from EoH results into heuristic.py
   (replacing the template tournament-selection body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
The baseline (tournament selection, tournament_size=3) is evaluated alongside
the EoH heuristic so performance gains are directly visible.

Algorithm context
-----------------
This corresponds to the `toolbox.select` step in DEAP's eaSimple algorithm.
The rest of the loop (SBX crossover, polynomial mutation, full generational
replacement) is fixed and identical for both baseline and EoH heuristic.
"""

import importlib
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation


def _load_select(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod.select


def _baseline_select(fitnesses: np.ndarray, k: int, tournament_size: int) -> np.ndarray:
    """Tournament selection — DEAP eaSimple default."""
    pop_size = len(fitnesses)
    selected = np.empty(k, dtype=int)
    for i in range(k):
        candidates = np.random.choice(pop_size, tournament_size, replace=False)
        selected[i] = candidates[np.argmin(fitnesses[candidates])]
    return selected


def _fmt_row(label: str, results: list[dict]) -> list[str]:
    lines = []
    scores = []
    for r in results:
        line = (f"  {label:35s} | {r['name']:12s} {r['dim']:3d}D | "
                f"mean: {r['mean']:12.4f}  std: {r['std']:10.4f}  "
                f"log1p: {r['log1p_mean']:.4f}")
        lines.append(line)
        scores.append(r['log1p_mean'])
    lines.append(f"  {'OVERALL ' + label:35s} | mean log1p score: {np.mean(scores):.4f}")
    return lines


if __name__ == "__main__":
    eva = Evaluation(
        pop_size=100, n_gen=200, tournament_size=3,
        cxpb=0.9, mutpb=0.1, eta_c=15.0, eta_m=20.0, n_runs=10,
    )

    print("Evaluating baseline (tournament selection, size=3) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_baseline_select)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_select = _load_select("heuristic")
    eoh_results = eva.evaluate(eoh_select)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    header = (f"  {'Heuristic':35s} | {'Function':12s} {'Dim':3s} | "
              f"{'Mean best':>14}  {'Std':>12}  {'log1p'}")
    sep = "-" * len(header)

    output_lines = [
        "eaSimple Parent Selection – Evaluation Results",
        "pop_size=100  n_gen=200  cxpb=0.9  mutpb=0.1  eta_c=15  eta_m=20  n_runs=10",
        sep, header, sep,
    ]
    output_lines += _fmt_row("Baseline (tournament, size=3)", baseline_results)
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
