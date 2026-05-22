"""Evaluate a (1+1)-ES mutation generator designed by EoH on the full benchmark suite.

Usage
-----
1. Copy the best `generate_mutation` function from EoH results into heuristic.py
   (replacing the adaptive mixed Gaussian–Cauchy body).
2. Run:   python runEval.py

Results are printed to the console and written to results.txt.
The BASELINE is nevergrad's actual OnePlusOne.minimize() so results are
directly comparable to the library's published performance.
"""

import importlib
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation


def _load_heuristic():
    mod = importlib.import_module('heuristic')
    return importlib.reload(mod).generate_mutation


def _fmt_row(label: str, results: list[dict]) -> list[str]:
    lines = []
    scores = []
    for r in results:
        line = (f"  {label:35s} | {r['name']:12s} {r['dim']:3d}D | "
                f"mean: {r['mean']:12.4f}  std: {r['std']:10.4f}  "
                f"log1p: {r['log1p_mean']:.4f}")
        lines.append(line)
        scores.append(r['log1p_mean'])
    lines.append(
        f"  {'OVERALL ' + label:35s} | mean log1p score: {np.mean(scores):.4f}"
    )
    return lines


if __name__ == '__main__':
    eva = Evaluation(max_evals=5000, n_runs=10)

    # Baseline: nevergrad's stock OnePlusOne (no modification)
    print('Evaluating baseline (nevergrad OnePlusOne.minimize()) ...')
    t0 = time.time()
    baseline_results = eva.evaluate_nevergrad_baseline()
    print(f'  Done in {time.time() - t0:.1f}s\n')

    # EoH variant: same nevergrad engine, evolved mutation step
    print('Evaluating EoH heuristic (heuristic.py via _EolOnePlusOne) ...')
    t0 = time.time()
    eoh_mutation = _load_heuristic()
    eoh_results = eva.evaluate(eoh_mutation)
    print(f'  Done in {time.time() - t0:.1f}s\n')

    header = (f"  {'Heuristic':35s} | {'Function':12s} {'Dim':3s} | "
              f"{'Mean best':>14}  {'Std':>12}  {'log1p'}")
    sep = '-' * len(header)

    output_lines = [
        '(1+1)-ES Mutation Generator – Evaluation Results',
        'nevergrad OnePlusOne engine  max_evals=5000  n_runs=10',
        sep, header, sep,
    ]
    output_lines += _fmt_row('nevergrad OnePlusOne (baseline)', baseline_results)
    output_lines += [sep]
    output_lines += _fmt_row('EoH heuristic (_EolOnePlusOne)', eoh_results)
    output_lines += [sep]

    output_lines.append(
        '\n  Per-benchmark improvement (EoH vs baseline, lower log1p = better):'
    )
    for b, e in zip(baseline_results, eoh_results):
        gain = b['log1p_mean'] - e['log1p_mean']
        tag = 'BETTER' if gain > 1e-6 else ('WORSE ' if gain < -1e-6 else 'TIE   ')
        output_lines.append(
            f'  {tag}  {e["name"]:12s} {e["dim"]:3d}D  '
            f'gain: {gain:+.4f}  '
            f'({b["mean"]:.4f} → {e["mean"]:.4f})'
        )

    full_output = '\n'.join(output_lines)
    print(full_output)

    results_path = os.path.join(os.path.dirname(__file__), 'results.txt')
    with open(results_path, 'w') as f:
        f.write(full_output + '\n')
    print(f'\nResults saved to {results_path}')
