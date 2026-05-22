"""Evaluate a sep-CMA-ES diagonal adaptation rule designed by EoH against the baseline.

Usage
-----
1. Copy the best `adapt_diagonal_cov` function from EoH results into heuristic.py
   (replacing the template body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
Both n=100 and n=200 variants of each benchmark are tested to assess
generalization across dimensionalities.
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
    return mod.adapt_diagonal_cov


def _sepcmaes_baseline(d, p_c, weights, y_k, c1, cmu, cc, hsig,
                       n, generation, max_generations):
    """Standard sep-CMA-ES rank-1 + rank-mu diagonal update."""
    rank1  = c1  * (p_c ** 2 + (1 - hsig) * cc * (2 - cc) * d)
    rankmu = cmu * np.einsum('i,ij->j', weights, y_k ** 2)
    return (1 - c1 - cmu) * d + rank1 + rankmu


def _fmt_results(label: str, results: list[dict]) -> list[str]:
    lines = []
    scores = []
    for r in results:
        lines.append(
            f"  {label:36s} | {r['name']:12s} {r['dim']:4d}-D | "
            f"mean: {r['mean']:14.4f}  std: {r['std']:12.4f}  "
            f"log1p: {r['log1p_mean']:.4f}"
        )
        scores.append(r['log1p_mean'])
    lines.append(
        f"  {'OVERALL ' + label:36s} | mean log1p score: {np.mean(scores):.4f}"
    )
    return lines


if __name__ == "__main__":
    eva = Evaluation(max_evals=60_000, n_runs=10)

    print("Evaluating baseline (standard sep-CMA-ES) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_sepcmaes_baseline)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_fn = _load_adapt_fn("heuristic")
    eoh_results = eva.evaluate(eoh_fn)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    cfg = "max_evals=60,000  n_runs=10  dims: 100-D & 200-D"
    header = (
        f"  {'Heuristic':36s} | {'Function':12s} {'Dim':5s} | "
        f"{'Mean best':>16}  {'Std':>14}  {'log1p'}"
    )
    sep = "-" * len(header)

    output_lines = [
        "Large-Scale sep-CMA-ES — Diagonal Variance Adaptation Evaluation",
        cfg, sep, header, sep,
    ]
    output_lines += _fmt_results("Baseline (sep-CMA-ES)", baseline_results)
    output_lines += [sep]
    output_lines += _fmt_results("EoH heuristic", eoh_results)
    output_lines += [sep]

    output_lines.append(
        "\n  Per-benchmark improvement (EoH vs sep-CMA-ES, lower log1p = better):"
    )
    for b, e in zip(baseline_results, eoh_results):
        gain = b['log1p_mean'] - e['log1p_mean']
        tag = "BETTER" if gain > 1e-6 else ("WORSE " if gain < -1e-6 else "TIE   ")
        output_lines.append(
            f"  {tag}  {e['name']:12s} {e['dim']:4d}-D  "
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
