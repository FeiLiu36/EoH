"""Evaluate a multi-objective metaheuristic designed by EoH against the baseline.

Usage
-----
1. Copy the best Metaheuristic class from EoH results into heuristic.py,
   replacing the placeholder body.
2. Run:  python runEval.py

baseline.py  — fixed reference algorithm (scalarized ES); do not modify.
heuristic.py — paste the EoH-designed Metaheuristic class here.

Results are printed to the console and written to results.txt.
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


def _fmt(label: str, results: list[dict]) -> list[str]:
    lines = []
    mean_hvs = []
    for r in results:
        lines.append(
            f"  {label:40s} | {r['name']:22s} | "
            f"HV: {r['mean_hv']:.4f} ± {r['std_hv']:.4f}  "
            f"PF size: {r['mean_pareto_size']:6.1f}"
        )
        mean_hvs.append(r['mean_hv'])
    lines.append(
        f"  {'OVERALL ' + label:40s} | mean HV: {np.mean(mean_hvs):.4f}"
    )
    return lines


if __name__ == "__main__":
    eva = Evaluation(dim=10, budget=5000, n_instances=4, n_runs=5)

    print("Evaluating baseline (baseline.py)  ...")
    t0 = time.time()
    baseline_cls = _load_class("baseline")
    baseline_results = eva.evaluate(baseline_cls)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH metaheuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_cls = _load_class("heuristic")
    eoh_results = eva.evaluate(eoh_cls)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    cfg = "dim=10  budget=5000  n_instances=4 (ZDT1-4)  n_runs=5  ref_pt=(1.1, 1.1)"
    header = (
        f"  {'Heuristic':40s} | {'Instance':22s} | "
        f"{'HV (mean ± std)':>22}  {'PF size':>8}"
    )
    sep = "-" * len(header)

    output_lines = [
        "Multi-Objective BBOB Metaheuristic — Evaluation Results",
        cfg, sep, header, sep,
    ]
    output_lines += _fmt("Baseline (NSGA-II)", baseline_results)
    output_lines += [sep]
    output_lines += _fmt("EoH metaheuristic", eoh_results)
    output_lines += [sep]

    b_mean = np.mean([r['mean_hv'] for r in baseline_results])
    e_mean = np.mean([r['mean_hv'] for r in eoh_results])
    gap_pct = (e_mean - b_mean) / max(b_mean, 1e-12) * 100

    output_lines.append(
        "\n  Per-instance HV improvement (EoH vs baseline, higher HV = better):"
    )
    for b, e in zip(baseline_results, eoh_results):
        gain = e['mean_hv'] - b['mean_hv']
        tag  = "BETTER" if gain > 1e-6 else ("WORSE " if gain < -1e-6 else "TIE   ")
        pct  = gain / max(b['mean_hv'], 1e-12) * 100
        output_lines.append(
            f"  {tag}  {e['name']:22s}  "
            f"ΔHV: {gain:+.4f} ({pct:+.2f}%)  "
            f"({b['mean_hv']:.4f} → {e['mean_hv']:.4f})"
        )
    output_lines.append(
        f"\n  Overall HV gain vs baseline: {gap_pct:+.2f}%  "
        f"({b_mean:.4f} → {e_mean:.4f})"
    )

    full_output = "\n".join(output_lines)
    print(full_output)

    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w") as fh:
        fh.write(full_output + "\n")
    print(f"\nResults saved to {results_path}")
