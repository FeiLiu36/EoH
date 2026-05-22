"""Evaluate an ES step-size adaptation rule designed by EoH on the full benchmark suite.

Usage
-----
1. Copy the best `adapt_step_size` function from EoH results into heuristic.py
   (replacing the template Rechenberg 1/5-success rule body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
The baseline (Rechenberg 1/5-success rule) is evaluated alongside
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


def _load_adapt_fn(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod.adapt_step_size


def _baseline_adapt(sigma, acceptance_rate, f_parent, f_offspring, n, generation, max_generations):
    """Rechenberg 1/5 success rule reference implementation."""
    c = 0.817
    if acceptance_rate > 0.2:
        return sigma / c
    elif acceptance_rate < 0.2:
        return sigma * c
    return sigma


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
    eva = Evaluation(lam=10, max_evals=15000, n_runs=10, ema_alpha=0.2)

    print("Evaluating baseline (Rechenberg 1/5 success rule) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_baseline_adapt)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_fn = _load_adapt_fn("heuristic")
    eoh_results = eva.evaluate(eoh_fn)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    header = (f"  {'Heuristic':35s} | {'Function':12s} {'Dim':3s} | "
              f"{'Mean best':>14}  {'Std':>12}  {'log1p'}")
    sep = "-" * len(header)

    output_lines = [
        "ES Step-Size Adaptation Rule – Evaluation Results",
        f"lam=10  max_evals=15000  n_runs=10  ema_alpha=0.2",
        sep, header, sep,
    ]
    output_lines += _fmt_row("Baseline (1/5 success rule)", baseline_results)
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
