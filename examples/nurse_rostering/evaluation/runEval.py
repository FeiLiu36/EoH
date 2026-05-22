"""Evaluate a nurse-rostering scoring function designed by EoH against the baseline.

Usage
-----
1. Copy the best `score_assignment` function from EoH results into heuristic.py
   (replacing the template body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
Both 14-day (2-week) and 21-day (3-week) rosters with 8 nurses are tested.
"""

import importlib
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation


def _load_score_fn(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod.score_assignment


def _baseline(nurse_idx, shift_type, day, nurse_workload, nurse_preferences,
              consecutive_days, last_shift_type, target_workload, n_days):
    """Classic linear scoring: preference - workload gap - consecutive - night→morning."""
    preference = nurse_preferences[nurse_idx, shift_type]
    workload_gap = nurse_workload[nurse_idx] - target_workload
    consecutive_penalty = max(0.0, float(consecutive_days[nurse_idx]) - 4.0)
    night_morning_penalty = 1.0 if (shift_type == 0 and last_shift_type[nurse_idx] == 2) else 0.0
    return float(preference - 0.5 * workload_gap - 2.0 * consecutive_penalty
                 - 5.0 * night_morning_penalty)


def _fmt_results(label: str, results: list[dict]) -> list[str]:
    lines = []
    composites = []
    for r in results:
        lines.append(
            f"  {label:36s} | {r['label']:26s} | "
            f"composite: {r['mean_composite']:6.3f} ± {r['std_composite']:.3f}  "
            f"wl_std: {r['mean_workload_std']:.3f}  "
            f"pref: {r['mean_preference_mean']:+.3f}  "
            f"consec: {r['mean_consec_violations']:.1f}  "
            f"nm: {r['mean_nm_violations']:.1f}"
        )
        composites.append(r['mean_composite'])
    lines.append(
        f"  {'OVERALL ' + label:36s} | mean composite: {np.mean(composites):.4f}"
    )
    return lines


if __name__ == "__main__":
    eva = Evaluation()

    print("Evaluating baseline (linear scoring) ...")
    t0 = time.time()
    baseline_results = eva.evaluate(_baseline)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Evaluating EoH heuristic (heuristic.py) ...")
    t0 = time.time()
    eoh_fn = _load_score_fn("heuristic")
    eoh_results = eva.evaluate(eoh_fn)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    cfg = "8 nurses | 14-day & 21-day horizons | 10 instances each"
    header = (
        f"  {'Heuristic':36s} | {'Config':26s} | "
        f"{'Composite':>14}  {'Wl-std':>8}  {'Pref':>6}  {'Consec':>7}  {'N→M':>5}"
    )
    sep = "-" * len(header)

    output_lines = [
        "Nurse Rostering — Shift-Assignment Scoring Function Evaluation",
        cfg, sep, header, sep,
    ]
    output_lines += _fmt_results("Baseline (linear)", baseline_results)
    output_lines += [sep]
    output_lines += _fmt_results("EoH heuristic", eoh_results)
    output_lines += [sep]

    output_lines.append(
        "\n  Per-group improvement (EoH vs Baseline, lower composite = better):"
    )
    for b, e in zip(baseline_results, eoh_results):
        gain = b['mean_composite'] - e['mean_composite']
        tag = "BETTER" if gain > 1e-4 else ("WORSE " if gain < -1e-4 else "TIE   ")
        output_lines.append(
            f"  {tag}  {e['label']:26s}  "
            f"gain: {gain:+.4f}  "
            f"({b['mean_composite']:.4f} → {e['mean_composite']:.4f})"
        )

    baseline_score = np.mean([r['mean_composite'] for r in baseline_results])
    eoh_score = np.mean([r['mean_composite'] for r in eoh_results])
    output_lines.append(
        f"\n  Overall composite: baseline {baseline_score:.4f}  →  "
        f"EoH {eoh_score:.4f}  (Δ {eoh_score - baseline_score:+.4f})"
    )

    full_output = "\n".join(output_lines)
    print(full_output)

    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w") as f:
        f.write(full_output + "\n")
    print(f"\nResults saved to {results_path}")
