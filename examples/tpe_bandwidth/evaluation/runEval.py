"""Evaluate a TPE observation-weighting rule designed by EoH on the full benchmark suite.

Usage
-----
1. Copy the best `compute_weights` function from EoH results into heuristic.py
   (replacing the template Optuna default_weights body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
The baseline (Optuna's built-in default_weights) is evaluated alongside the EoH
heuristic so performance gains are directly visible.

Both the baseline and EoH heuristic are passed directly to
`optuna.samplers.TPESampler(weights=...)`, so they run through Optuna's actual
TPE implementation with no reimplementation.
"""

import importlib
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation


def _load_weights_fn(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod.compute_weights


def _baseline_weights(n: int) -> np.ndarray:
    """Optuna's built-in default_weights reference implementation."""
    if n == 0:
        return np.array([])
    elif n < 25:
        return np.ones(n)
    else:
        ramp = np.linspace(1.0 / n, 1.0, num=n - 25)
        flat = np.ones(25)
        return np.concatenate([ramp, flat])


def _fmt_row(label: str, results: list[dict]) -> list[str]:
    lines = []
    scores = []
    for r in results:
        domain = f"[{r['lo']:.2f}, {r['hi']:.2f}]"
        line = (f"  {label:35s} | {r['name']:10s} {domain:22s} | "
                f"mean: {r['mean']:12.6f}  std: {r['std']:10.6f}  "
                f"log1p: {r['log1p_mean']:.4f}")
        lines.append(line)
        scores.append(r['log1p_mean'])
    lines.append(f"  {'OVERALL ' + label:35s} | mean log1p score: {np.mean(scores):.4f}")
    return lines


if __name__ == "__main__":
    eva = Evaluation(n_startup=20, n_iter=60, n_ei_candidates=64, n_runs=10)

    print("Evaluating baseline (Optuna default_weights) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_baseline_weights)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_fn = _load_weights_fn("heuristic")
    eoh_results = eva.evaluate(eoh_fn)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    header = (f"  {'Heuristic':35s} | {'Function':10s} {'Domain':22s} | "
              f"{'Mean best':>14}  {'Std':>12}  {'log1p'}")
    sep = "-" * len(header)

    output_lines = [
        "TPE Sampler Observation Weights – Evaluation Results",
        "n_startup=20  n_iter=60  n_ei_candidates=64  n_runs=10",
        "Harness: optuna.samplers.TPESampler(weights=...)",
        sep, header, sep,
    ]
    output_lines += _fmt_row("Baseline (default_weights)", baseline_results)
    output_lines += [sep]
    output_lines += _fmt_row("EoH heuristic", eoh_results)
    output_lines += [sep]

    output_lines.append("\n  Per-benchmark improvement (EoH vs baseline, lower log1p = better):")
    for b, e in zip(baseline_results, eoh_results):
        gain = b['log1p_mean'] - e['log1p_mean']
        tag = "BETTER" if gain > 1e-6 else ("WORSE " if gain < -1e-6 else "TIE   ")
        domain = f"[{e['lo']:.2f}, {e['hi']:.2f}]"
        output_lines.append(
            f"  {tag}  {e['name']:10s} {domain:22s}  "
            f"gain: {gain:+.4f}  "
            f"({b['mean']:.6f} → {e['mean']:.6f})"
        )

    full_output = "\n".join(output_lines)
    print(full_output)

    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w") as f:
        f.write(full_output + "\n")
    print(f"\nResults saved to {results_path}")
