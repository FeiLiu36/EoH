"""Evaluate a complete EoH-designed metaheuristic against the (1+lambda)-ES baseline.

Usage
-----
1. Copy the best `Metaheuristic` class from EoH results into heuristic.py
   (replacing the template body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
Both 10-D and 20-D variants of each benchmark are tested to assess generalization.
"""

import importlib
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation


def _load_class(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod.Metaheuristic


class _BaselineMetaheuristic:
    """(1+lambda)-ES with Rechenberg 1/5-rule step-size adaptation."""

    def __init__(self, func, dim, bounds, budget):
        self.func   = func
        self.dim    = dim
        self.lo     = bounds[0].copy()
        self.hi     = bounds[1].copy()
        self.budget = budget

    def solve(self):
        lam = max(4, 4 + int(3 * np.log(max(self.dim, 1))))
        sigma = float((self.hi - self.lo).mean()) / 4.0

        x_parent = self.lo + (self.hi - self.lo) * np.random.rand(self.dim)
        f_parent = self.func(x_parent)
        n_evals = 1

        while n_evals < self.budget:
            actual_lam = min(lam, self.budget - n_evals)
            offspring = [
                np.clip(x_parent + np.random.randn(self.dim) * sigma, self.lo, self.hi)
                for _ in range(actual_lam)
            ]
            successes = 0
            for x in offspring:
                f = self.func(x)
                n_evals += 1
                if f < f_parent:
                    f_parent = f
                    x_parent = x.copy()
                    successes += 1

            rate = successes / actual_lam
            c = 0.817
            if rate > 0.2:
                sigma /= c
            elif rate < 0.2:
                sigma *= c
            sigma = float(np.clip(sigma, 1e-12, float((self.hi - self.lo).mean())))

        return x_parent


def _fmt_results(label: str, results: list[dict]) -> list[str]:
    lines = []
    scores = []
    for r in results:
        lines.append(
            f"  {label:36s} | {r['name']:12s} {r['dim']:3d}-D | "
            f"mean: {r['mean']:14.4f}  std: {r['std']:12.4f}  "
            f"log1p: {r['log1p_mean']:.4f}"
        )
        scores.append(r['log1p_mean'])
    lines.append(
        f"  {'OVERALL ' + label:36s} | mean log1p score: {np.mean(scores):.4f}"
    )
    return lines


if __name__ == "__main__":
    eva = Evaluation(budget=10000, n_runs=10)

    print("Evaluating baseline ((1+lambda)-ES with 1/5-rule) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_BaselineMetaheuristic)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH metaheuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_class = _load_class("heuristic")
    eoh_results = eva.evaluate(eoh_class)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    cfg = "budget=10000  n_runs=10  dims: 10-D & 20-D"
    header = (
        f"  {'Metaheuristic':36s} | {'Function':12s} {'Dim':4s} | "
        f"{'Mean best':>16}  {'Std':>14}  {'log1p'}"
    )
    sep = "-" * len(header)

    output_lines = [
        "BBOB Metaheuristic Design — Full Evaluation",
        cfg, sep, header, sep,
    ]
    output_lines += _fmt_results("Baseline ((1+lam)-ES)", baseline_results)
    output_lines += [sep]
    output_lines += _fmt_results("EoH metaheuristic", eoh_results)
    output_lines += [sep]

    output_lines.append(
        "\n  Per-benchmark improvement (EoH vs baseline, lower log1p = better):"
    )
    for b, e in zip(baseline_results, eoh_results):
        gain = b['log1p_mean'] - e['log1p_mean']
        tag = "BETTER" if gain > 1e-6 else ("WORSE " if gain < -1e-6 else "TIE   ")
        output_lines.append(
            f"  {tag}  {e['name']:12s} {e['dim']:3d}-D  "
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
