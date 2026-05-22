"""Evaluate a dynamic-EA response strategy designed by EoH.

Usage
-----
1. Copy the best `respond_to_change` function from EoH results into heuristic.py
   (replacing the elite+immigrants body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
The baseline (hypermutation) is evaluated alongside the EoH heuristic
so performance gains are directly visible.
"""

import importlib
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation


def _baseline_respond(population, fitness, best_position, bounds):
    """Hypermutation baseline: perturb all individuals with large Gaussian noise."""
    sigma = (bounds[1] - bounds[0]).mean() * 0.1
    new_pop = population + np.random.normal(0.0, sigma, population.shape)
    return np.clip(new_pop, bounds[0], bounds[1])


def _load_heuristic():
    mod = importlib.import_module('heuristic')
    return importlib.reload(mod).respond_to_change


def _fmt_results(label: str, results: list[dict]) -> list[str]:
    lines = []
    all_errors = []
    for r in results:
        lines.append(
            f"  {label:35s} | {r['label']:15s} | "
            f"mean error: {r['mean_error']:7.4f}  std: {r['std_error']:6.4f}"
        )
        all_errors.append(r['mean_error'])
    lines.append(f"  {'OVERALL ' + label:35s} | mean tracking error: {np.mean(all_errors):.4f}")
    return lines


if __name__ == '__main__':
    eva = Evaluation(n_test=16, pop_size=30, k_iter=50)

    print('Evaluating baseline (hypermutation) ...')
    t0 = time.time()
    baseline_results = eva.evaluate(_baseline_respond)
    print(f'  Done in {time.time() - t0:.1f}s\n')

    print('Evaluating EoH heuristic (heuristic.py) ...')
    t0 = time.time()
    eoh_respond = _load_heuristic()
    eoh_results = eva.evaluate(eoh_respond)
    print(f'  Done in {time.time() - t0:.1f}s\n')

    header = (f"  {'Heuristic':35s} | {'Scenario':15s} | "
              f"{'Mean error':>14}  {'Std':>10}")
    sep = '-' * len(header)

    output_lines = [
        'Dynamic-EA Response Strategy – Evaluation Results',
        f'n_test=16  pop_size=30  k_iter=50',
        sep, header, sep,
    ]
    output_lines += _fmt_results('Baseline (hypermutation)', baseline_results)
    output_lines += [sep]
    output_lines += _fmt_results('EoH heuristic', eoh_results)
    output_lines += [sep]

    output_lines.append('\n  Per-scenario improvement (EoH vs baseline, lower error = better):')
    for b, e in zip(baseline_results, eoh_results):
        gain = b['mean_error'] - e['mean_error']
        tag = 'BETTER' if gain > 1e-6 else ('WORSE ' if gain < -1e-6 else 'TIE   ')
        output_lines.append(
            f'  {tag}  {e["label"]:15s}  '
            f'gain: {gain:+.4f}  '
            f'({b["mean_error"]:.4f} → {e["mean_error"]:.4f})'
        )

    full_output = '\n'.join(output_lines)
    print(full_output)

    results_path = os.path.join(os.path.dirname(__file__), 'results.txt')
    with open(results_path, 'w') as f:
        f.write(full_output + '\n')
    print(f'\nResults saved to {results_path}')
