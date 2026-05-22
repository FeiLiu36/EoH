"""Evaluate a Bayesian Optimisation acquisition function discovered by EoH.

Usage
-----
1. Copy the best `acquisition` function from the EoH results
   (e.g. results/samples/samples_best.json) into heuristic.py,
   replacing the LCB template body.
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
Four baselines are evaluated alongside the EoH heuristic:
  - LCB  (Lower Confidence Bound, kappa=2)
  - EI   (Expected Improvement)
  - PI   (Probability of Improvement)
  - UCB  (Upper Confidence Bound — maximisation orientation, kappa=2)
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
# Baseline acquisition functions  (all: higher score → evaluate next)
# ------------------------------------------------------------------

def _lcb(mu: np.ndarray, sigma: np.ndarray, f_best: float,
         kappa: float = 2.0) -> np.ndarray:
    """Lower Confidence Bound (minimisation form)."""
    return -mu + kappa * sigma


def _ei(mu: np.ndarray, sigma: np.ndarray, f_best: float,
        xi: float = 0.01) -> np.ndarray:
    """Expected Improvement."""
    from scipy.stats import norm
    imp = f_best - mu - xi
    z = imp / (sigma + 1e-9)
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    return np.where(sigma < 1e-9, 0.0, ei)


def _pi(mu: np.ndarray, sigma: np.ndarray, f_best: float,
        xi: float = 0.01) -> np.ndarray:
    """Probability of Improvement."""
    from scipy.stats import norm
    z = (f_best - mu - xi) / (sigma + 1e-9)
    return norm.cdf(z)


def _lcb_adaptive(mu: np.ndarray, sigma: np.ndarray, f_best: float) -> np.ndarray:
    """LCB with kappa=5 (more exploratory)."""
    return -mu + 5.0 * sigma


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _fmt_rows(label: str, results: list[dict]) -> list[str]:
    lines = []
    log_vals = []
    for r in results:
        lines.append(
            f"  {label:35s} | {r['name']:10s} {r['n_var']}D | "
            f"log10(regret): {r['log_regret_mean']:7.3f} ± {r['log_regret_std']:.3f}  "
            f"regret: {r['regret_mean']:.4e} ± {r['regret_std']:.4e}"
        )
        log_vals.append(r['log_regret_mean'])
    lines.append(
        f"  {'OVERALL ' + label:35s} | mean log10(regret): {np.mean(log_vals):.3f}"
    )
    return lines


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    eva = Evaluation(n_init=10, n_iter=40, n_candidates=512, n_runs=20)

    baselines = [
        ("LCB (kappa=2)",       _lcb),
        ("EI  (xi=0.01)",       _ei),
        ("PI  (xi=0.01)",       _pi),
        ("LCB adaptive (k=5)",  _lcb_adaptive),
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
    all_results["EoH heuristic"] = eva.evaluate(mod.acquisition)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    header = (f"  {'Acquisition function':35s} | {'Benchmark':14s} | "
              f"{'log10(regret)':>16}  {'regret':>12}")
    sep = "-" * len(header)

    output_lines = [
        "Bayesian Optimisation Acquisition Function – Evaluation Results",
        f"n_init={eva.n_init}  n_iter={eva.n_iter}  "
        f"n_candidates={eva.n_candidates}  n_runs={eva.n_runs}",
        sep, header, sep,
    ]
    for name, results in all_results.items():
        output_lines += _fmt_rows(name, results)
        output_lines.append(sep)

    # Per-instance gain vs LCB baseline
    lcb_res = all_results["LCB (kappa=2)"]
    eoh_res = all_results["EoH heuristic"]
    output_lines.append(
        "\n  Per-instance improvement: EoH vs LCB (lower log10(regret) = better)"
    )
    for b, e in zip(lcb_res, eoh_res):
        gain = b['log_regret_mean'] - e['log_regret_mean']
        tag = "BETTER" if gain > 0.05 else ("WORSE " if gain < -0.05 else "TIE   ")
        output_lines.append(
            f"  {tag}  {e['name']:10s} {e['n_var']}D  "
            f"gain: {gain:+.3f} log units  "
            f"({b['log_regret_mean']:.3f} → {e['log_regret_mean']:.3f})"
        )

    full_output = "\n".join(output_lines)
    print(full_output)

    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w") as f:
        f.write(full_output + "\n")
    print(f"\nResults saved to {results_path}")
