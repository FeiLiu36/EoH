"""Evaluate a MOEA/D decomposition operator discovered by EoH.

Usage
-----
1. Copy the best `custom_decomposition` function from the EoH results
   (e.g. results/samples/samples_best.json) into heuristic.py,
   replacing the Tchebycheff template body.
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
Three baselines are evaluated alongside the EoH heuristic:
  - Tchebycheff  (weighted Chebyshev distance from ideal point)
  - Weighted Sum (linear aggregation)
  - PBI          (Penalty-Based Boundary Intersection, theta=5)
"""

import importlib
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation


# ------------------------------------------------------------------
# Baseline decomposition functions
# ------------------------------------------------------------------

def _tchebycheff(F: np.ndarray, weights: np.ndarray,
                 ideal_point: np.ndarray) -> np.ndarray:
    return np.max(np.abs(F - ideal_point) * weights, axis=1)


def _weighted_sum(F: np.ndarray, weights: np.ndarray,
                  ideal_point: np.ndarray) -> np.ndarray:
    return np.sum(weights * F, axis=1)


def _pbi(F: np.ndarray, weights: np.ndarray,
         ideal_point: np.ndarray, theta: float = 5.0) -> np.ndarray:
    diff = F - ideal_point                                  # (n, m)
    w_norm = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-12
    w_unit = weights / w_norm
    d1 = np.sum(diff * w_unit, axis=1)                     # projection length
    d2 = np.linalg.norm(diff - d1[:, np.newaxis] * w_unit, axis=1)
    return d1 + theta * d2


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _fmt_rows(label: str, results: list[dict]) -> list[str]:
    lines = []
    hv_values = []
    for r in results:
        line = (f"  {label:35s} | {r['name']:6s} {r['n_var']:2d}var {r['n_obj']}obj | "
                f"HV mean: {r['hv_mean']:.4f}  std: {r['hv_std']:.4f}")
        lines.append(line)
        hv_values.append(r['hv_mean'])
    lines.append(f"  {'OVERALL ' + label:35s} | mean HV: {np.mean(hv_values):.4f}")
    return lines


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    eva = Evaluation(n_gen=200, n_runs=10, T=5)

    baselines = [
        ("Tchebycheff (baseline)", _tchebycheff),
        ("Weighted Sum",           _weighted_sum),
        ("PBI (theta=5)",          _pbi),
    ]

    all_results: dict[str, list[dict]] = {}

    for name, fn in baselines:
        print(f"Evaluating {name} ...")
        t0 = time.time()
        all_results[name] = eva.evaluate(fn)
        print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    heuristic_mod = importlib.import_module("heuristic")
    heuristic_mod = importlib.reload(heuristic_mod)
    eoh_fn = heuristic_mod.custom_decomposition
    all_results["EoH heuristic"] = eva.evaluate(eoh_fn)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    header = (f"  {'Decomposition':35s} | {'Problem':20s} | "
              f"{'HV mean':>10}  {'HV std':>8}")
    sep = "-" * len(header)

    output_lines = [
        "MOEA/D Decomposition Operator – Evaluation Results",
        f"n_gen={eva.n_gen}  n_runs={eva.n_runs}  T={eva.T}  "
        f"hv_samples={eva.hv_samples}",
        sep, header, sep,
    ]

    for name, results in all_results.items():
        output_lines += _fmt_rows(name, results)
        output_lines.append(sep)

    # Per-instance improvement vs Tchebycheff baseline
    tcheby_results = all_results["Tchebycheff (baseline)"]
    eoh_results = all_results["EoH heuristic"]
    output_lines.append(
        "\n  Per-instance improvement: EoH vs Tchebycheff (higher HV = better)"
    )
    for b, e in zip(tcheby_results, eoh_results):
        gain = e['hv_mean'] - b['hv_mean']
        tag = "BETTER" if gain > 1e-4 else ("WORSE " if gain < -1e-4 else "TIE   ")
        output_lines.append(
            f"  {tag}  {e['name']:6s} {e['n_var']:2d}var  "
            f"gain: {gain:+.4f}  ({b['hv_mean']:.4f} → {e['hv_mean']:.4f})"
        )

    full_output = "\n".join(output_lines)
    print(full_output)

    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w") as f:
        f.write(full_output + "\n")
    print(f"\nResults saved to {results_path}")
