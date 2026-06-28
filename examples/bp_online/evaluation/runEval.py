# -------
# Evaluation code for EoH on Online Bin Packing
# --------
# Benchmarks a set of scoring heuristics on the Weibull test datasets with
# 1k / 5k / 10k items, each under bin capacity 100 and 500. Reports the excess
# over the L1 lower bound (lower is better) as a method x configuration table.
#
# More results may refer to
# Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, Qingfu Zhang
# "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model"
# ICML 2024, https://arxiv.org/abs/2401.02051.

import importlib
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BP_ROOT = os.path.join(HERE, '..')
DATA_DIR = os.path.join(BP_ROOT, 'testingdata')

# Make get_instance / evaluation and the local heuristic modules importable.
sys.path.insert(0, BP_ROOT)
sys.path.insert(0, HERE)

from get_instance import GetData
from evaluation import Evaluation


def load_raw(pkl_name):
    """Load the raw item lists from a Weibull test pickle."""
    with open(os.path.join(DATA_DIR, pkl_name), 'rb') as f:
        raw = pickle.load(f)
    item_lists = []
    for _num_items, instances in raw.items():
        item_lists.extend(list(items) for items in instances)
    return item_lists


def build_dataset(item_lists, capacity):
    """Wrap raw item lists into the {instance_name: {...}} format for a capacity."""
    return {
        f'test_{i}': {
            'capacity': capacity,
            'num_items': len(items),
            'items': items,
        }
        for i, items in enumerate(item_lists, 1)
    }


# Heuristics to benchmark: (module name in this directory, display name).
HEURISTICS = [
    ('heuristic_best_fit', 'Best Fit'),
    ('heuristic_first_fit', 'First Fit'),
    ('heuristic_original_EoH_paper', 'EoH (original paper)'),
    ('heuristic_EoH_by_this_code_poor_generalization', 'EoH (this code poor gen.)'),
    ('heuristic_EoH_by_this_code', 'EoH (this code)'),
]

# Include the freshly evolved heuristic too, if one has been exported.
if os.path.exists(os.path.join(HERE, 'heuristic.py')):
    HEURISTICS.append(('heuristic', 'EoH (evolved)'))

# Test datasets (size label -> pickle) and bin capacities to sweep.
SIZES = [('1k', 'test_dataset_1k.pkl'),
         ('5k', 'test_dataset_5k.pkl'),
         ('10k', 'test_dataset_10k.pkl')]
CAPACITIES = [100, 500]

# Column order: all sizes at c100, then all sizes at c500.
COLUMNS = [f'{size}_c{cap}' for cap in CAPACITIES for size, _ in SIZES]

# Pre-load raw items once per size.
RAW = {size: load_raw(pkl) for size, pkl in SIZES}

eva = Evaluation()
gd = GetData()

# results[display_name][column] = excess fraction
results = {disp: {} for _, disp in HEURISTICS}
loaded = []
for mod_name, disp in HEURISTICS:
    try:
        heuristic = importlib.import_module(mod_name)
    except Exception as e:  # noqa: BLE001
        print(f'  [skip] {mod_name}: {e}')
        continue
    loaded.append(disp)
    for cap in CAPACITIES:
        for size, _ in SIZES:
            dataset = build_dataset(RAW[size], cap)
            avg_num_bins = -eva.evaluateGreedy(dataset, heuristic)
            lb = gd.l1_bound_dataset(dataset)
            results[disp][f'{size}_c{cap}'] = (avg_num_bins - lb) / lb
            print(f'  computed {disp:14} {size:>3}_c{cap}: '
                  f'{100 * results[disp][f"{size}_c{cap}"]:.2f}%')


def render_table():
    name_w = max(len('Method'), *(len(d) for d in loaded)) if loaded else len('Method')
    col_w = max(8, max(len(c) for c in COLUMNS))
    head = f'{"Method":<{name_w}} | ' + ' | '.join(f'{c:>{col_w}}' for c in COLUMNS)
    sep = '-' * len(head)
    rows = [head, sep]
    for disp in loaded:
        cells = []
        for c in COLUMNS:
            v = results[disp].get(c)
            cells.append(f'{100 * v:>{col_w - 1}.2f}%' if v is not None else f'{"-":>{col_w}}')
        rows.append(f'{disp:<{name_w}} | ' + ' | '.join(cells))
    return '\n'.join(rows)


table = ('\nExcess over L1 lower bound (%, lower is better)\n'
         '5 instances per cell; items ~ Weibull, sizes = #items, c = bin capacity\n\n'
         + render_table() + '\n')

print(table)
results_path = os.path.join(HERE, 'results.txt')
with open(results_path, 'w') as out:
    out.write(table)
print(f'Results written to {results_path}')
