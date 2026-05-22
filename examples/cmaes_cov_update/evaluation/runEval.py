"""Evaluate a CMA-ES covariance update rule designed by EoH on the full benchmark suite.

Usage
-----
1. Copy the best `update_covariance` function from EoH results into heuristic.py
   (replacing the template rank-1 + rank-mu body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
The standard CMA-ES baseline is evaluated alongside the EoH heuristic so
performance gains are directly visible.
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
    return mod.update_covariance


def _baseline_update(C, p_c, weights, y_k, c1, cmu, cc, hsig, n):
    """Standard CMA-ES rank-1 + rank-mu covariance update."""
    rank1 = c1 * (np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * C)
    rankmu = cmu * np.sum(
        [weights[i] * np.outer(y_k[i], y_k[i]) for i in range(len(weights))], axis=0
    )
    return (1 - c1 - cmu) * C + rank1 + rankmu


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
    eva = Evaluation(max_evals=10000, n_runs=10)

    print("Evaluating baseline (standard rank-1 + rank-mu) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_baseline_update)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_fn = _load_update_fn("heuristic")
    eoh_results = eva.evaluate(eoh_fn)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    header = (f"  {'Heuristic':35s} | {'Function':12s} {'Dim':3s} | "
              f"{'Mean best':>14}  {'Std':>12}  {'log1p'}")
    sep = "-" * len(header)

    output_lines = [
        "CMA-ES Covariance Update Rule – Evaluation Results",
        f"max_evals=10000  n_runs=10",
        sep, header, sep,
    ]
    output_lines += _fmt_row("Baseline (rank-1 + rank-mu)", baseline_results)
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
