"""Evaluate an NSGA-II crossover operator discovered by EoH (pymoo backend).

Usage
-----
1. Copy the best `crossover` function from the EoH results
   (e.g. results/samples/samples_best.json) into heuristic.py,
   replacing the SBX template body.
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
Three baselines are evaluated alongside the EoH heuristic:
  - SBX      (Simulated Binary Crossover, eta=15 — standard NSGA-II)
  - BLX-0.5  (Blend crossover with alpha=0.5)
  - DE/rand  (Differential-evolution-style recombination, F=0.5)
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
# Baseline crossover operators
# ------------------------------------------------------------------

def _sbx(x1: np.ndarray, x2: np.ndarray, eta: float = 15.0) -> tuple:
    """Simulated Binary Crossover."""
    c1, c2 = x1.copy(), x2.copy()
    for i in range(len(x1)):
        if np.random.random() < 0.5 and abs(x1[i] - x2[i]) > 1e-10:
            u = np.random.random()
            beta = (2 * u) ** (1.0 / (eta + 1)) if u <= 0.5 \
                else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))
            c1[i] = 0.5 * ((x1[i] + x2[i]) - beta * abs(x2[i] - x1[i]))
            c2[i] = 0.5 * ((x1[i] + x2[i]) + beta * abs(x2[i] - x1[i]))
    return c1, c2


def _blx(x1: np.ndarray, x2: np.ndarray, alpha: float = 0.5) -> tuple:
    """Blend Crossover (BLX-alpha)."""
    lo = np.minimum(x1, x2) - alpha * np.abs(x1 - x2)
    hi = np.maximum(x1, x2) + alpha * np.abs(x1 - x2)
    c1 = lo + np.random.rand(len(x1)) * (hi - lo)
    c2 = lo + np.random.rand(len(x1)) * (hi - lo)
    return c1, c2


def _de_rand(x1: np.ndarray, x2: np.ndarray, F: float = 0.5) -> tuple:
    """DE/rand-inspired recombination: one offspring via differential perturbation."""
    diff = x2 - x1
    c1 = x1 + F * diff
    c2 = x2 - F * diff
    return c1, c2


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _fmt_rows(label: str, results: list[dict]) -> list[str]:
    lines = []
    hv_vals = []
    for r in results:
        line = (f"  {label:40s} | {r['name']:5s} {r['n_var']:2d}var | "
                f"HV mean: {r['hv_mean']:.5f}  std: {r['hv_std']:.5f}")
        lines.append(line)
        hv_vals.append(r['hv_mean'])
    lines.append(f"  {'OVERALL ' + label:40s} | mean HV: {np.mean(hv_vals):.5f}")
    return lines


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    eva = Evaluation(pop_size=100, n_gen=200, n_runs=10)

    baselines = [
        ("SBX (NSGA-II baseline, eta=15)", _sbx),
        ("BLX-0.5",                        _blx),
        ("DE/rand (F=0.5)",                _de_rand),
    ]

    all_results: dict[str, list[dict]] = {}

    for name, fn in baselines:
        print(f"Evaluating {name} ...")
        t0 = time.time()
        all_results[name] = eva.evaluate(fn)
        print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    mod = importlib.import_module("heuristic")
    mod = importlib.reload(mod)
    all_results["EoH heuristic"] = eva.evaluate(mod.crossover)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    header = (f"  {'Crossover operator':40s} | {'Problem':12s} | "
              f"{'HV mean':>10}  {'HV std':>8}")
    sep = "-" * len(header)

    output_lines = [
        "NSGA-II Crossover Operator – Evaluation Results (pymoo backend)",
        f"pop_size={eva.pop_size}  n_gen={eva.n_gen}  n_runs={eva.n_runs}  "
        f"mutation=PM(eta=20)",
        sep, header, sep,
    ]
    for name, results in all_results.items():
        output_lines += _fmt_rows(name, results)
        output_lines.append(sep)

    # Per-instance gain vs SBX baseline
    sbx_res = all_results["SBX (NSGA-II baseline, eta=15)"]
    eoh_res = all_results["EoH heuristic"]
    output_lines.append(
        "\n  Per-instance improvement: EoH vs SBX (higher HV = better)"
    )
    for b, e in zip(sbx_res, eoh_res):
        gain = e['hv_mean'] - b['hv_mean']
        tag = "BETTER" if gain > 1e-4 else ("WORSE " if gain < -1e-4 else "TIE   ")
        output_lines.append(
            f"  {tag}  {e['name']:5s} {e['n_var']:2d}var  "
            f"gain: {gain:+.5f}  ({b['hv_mean']:.5f} → {e['hv_mean']:.5f})"
        )

    full_output = "\n".join(output_lines)
    print(full_output)

    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w") as f:
        f.write(full_output + "\n")
    print(f"\nResults saved to {results_path}")
