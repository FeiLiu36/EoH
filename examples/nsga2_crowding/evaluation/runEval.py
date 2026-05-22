"""Evaluate an NSGA-II crowding-distance operator discovered by EoH.

Usage
-----
1. Copy the best `crowding_distance` function from the EoH results
   (e.g. results/samples/samples_best.json) into heuristic.py,
   replacing the standard crowding-distance template body.
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
Three baselines are evaluated alongside the EoH heuristic:
  - Crowding Distance  (standard NSGA-II, Deb et al. 2002)
  - Nearest Neighbour  (min distance to any other solution in F-space)
  - HV Contribution    (approximate marginal hypervolume contribution)
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
# Baseline diversity metrics
# ------------------------------------------------------------------

def _crowding_distance(F: np.ndarray) -> np.ndarray:
    """Standard NSGA-II crowding distance."""
    n, m = F.shape
    dist = np.zeros(n)
    for obj in range(m):
        idx = np.argsort(F[:, obj])
        dist[idx[0]] = np.inf
        dist[idx[-1]] = np.inf
        f_range = F[idx[-1], obj] - F[idx[0], obj]
        if f_range < 1e-10:
            continue
        for k in range(1, n - 1):
            dist[idx[k]] += (F[idx[k + 1], obj] - F[idx[k - 1], obj]) / f_range
    return dist


def _nearest_neighbour(F: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distance: diversity = min distance to any other solution."""
    dists = np.sum((F[:, np.newaxis] - F[np.newaxis]) ** 2, axis=2)
    np.fill_diagonal(dists, np.inf)
    return np.sqrt(np.min(dists, axis=1))


def _hv_contribution(F: np.ndarray) -> np.ndarray:
    """Approximate hypervolume contribution using a 2D sweep.

    Each solution's contribution is the area it uniquely dominates
    (i.e. HV(F) - HV(F without solution i)).  Boundary solutions get inf.
    """
    n, m = F.shape
    if m != 2:
        # Fall back to crowding distance for non-2D fronts
        return _crowding_distance(F)

    ref = np.max(F, axis=0) * 1.1 + 1e-6

    def hv2d(pts):
        if len(pts) == 0:
            return 0.0
        sf = pts[np.argsort(pts[:, 0])]
        nd, prev_f2 = [], np.inf
        for p in sf:
            if p[1] < prev_f2:
                nd.append(p)
                prev_f2 = p[1]
        nd = np.array(nd)
        total = 0.0
        for i in range(len(nd)):
            f1_next = nd[i + 1, 0] if i + 1 < len(nd) else ref[0]
            total += (f1_next - nd[i, 0]) * (ref[1] - nd[i, 1])
        return total

    total_hv = hv2d(F)
    contrib = np.zeros(n)
    for i in range(n):
        rest = np.delete(F, i, axis=0)
        contrib[i] = total_hv - hv2d(rest)
    return contrib


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
        ("Crowding Distance (NSGA-II baseline)", _crowding_distance),
        ("Nearest Neighbour",                    _nearest_neighbour),
        ("HV Contribution",                      _hv_contribution),
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
    all_results["EoH heuristic"] = eva.evaluate(mod.crowding_distance)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    header = (f"  {'Diversity metric':40s} | {'Problem':12s} | "
              f"{'HV mean':>10}  {'HV std':>8}")
    sep = "-" * len(header)

    output_lines = [
        "NSGA-II Crowding Operator – Evaluation Results",
        f"pop_size={eva.pop_size}  n_gen={eva.n_gen}  n_runs={eva.n_runs}",
        sep, header, sep,
    ]
    for name, results in all_results.items():
        output_lines += _fmt_rows(name, results)
        output_lines.append(sep)

    # Per-instance gain vs standard crowding distance
    baseline_res = all_results["Crowding Distance (NSGA-II baseline)"]
    eoh_res = all_results["EoH heuristic"]
    output_lines.append(
        "\n  Per-instance improvement: EoH vs Crowding Distance (higher HV = better)"
    )
    for b, e in zip(baseline_res, eoh_res):
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
