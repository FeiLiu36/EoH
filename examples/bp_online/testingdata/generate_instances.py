# Generate the online bin-packing test datasets used by the bp_online example.
#
# Item sizes are drawn from 45 * Weibull(shape=3), clipped to [1, capacity] and
# rounded to integers (mean item size ~40 for capacity 100). Each dataset is a
# dict {num_items: [[items_instance_1], [items_instance_2], ...]}.
#
# Datasets are named "<N>k" where N is the item count in thousands
# (e.g. test_dataset_10k.pkl holds instances of 10,000 items each).
#
# A fixed per-dataset seed makes the committed pickles exactly reproducible:
# running this script regenerates byte-identical files.

import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Bins have this capacity; item sizes are clipped to [1, CAPACITY].
CAPACITY = 100
# Base seed; each dataset uses BASE_SEED + its index so files are independently
# reproducible (adding/removing a dataset does not change the others).
BASE_SEED = 2024

# (filename, num_instances, num_items)
DATASETS = [
    ('training_dataset_5k.pkl', 5, 5000),
    ('test_dataset_1k.pkl',     5, 1000),
    ('test_dataset_2k.pkl',     5, 2000),
    ('test_dataset_5k.pkl',     5, 5000),
    ('test_dataset_10k.pkl',    5, 10000),
    ('test_dataset_100k.pkl',   1, 100000),
]


def generate_weibull_dataset(num_instances, num_items, clipping_limit=CAPACITY, seed=None):
    """Generate `num_instances` instances of `num_items` Weibull-distributed items.

    Returns {num_items: [[items], ...]} with integer item sizes in
    [1, clipping_limit].
    """
    rng = np.random.default_rng(seed)
    dataset = {num_items: []}
    for _ in range(num_instances):
        samples = rng.weibull(3, num_items) * 45        # 45 * Weibull(shape=3)
        samples = np.clip(samples, 1, clipping_limit)   # keep sizes in [1, cap]
        sizes = np.round(samples).astype(int)
        dataset[num_items].append(sizes.tolist())
    return dataset


def read_dataset_from_file(filename, capacity=CAPACITY):
    """Load a test pickle into the {"Weibull Nk": {test_i: {...}}} format."""
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    transformed = {}
    for num_items, instances in dataset.items():
        label = f"Weibull {num_items // 1000}k"
        transformed[label] = {
            f"test_{i}": {
                "capacity": capacity,
                "num_items": num_items,
                "items": items,
            }
            for i, items in enumerate(instances, 1)
        }
    return transformed


def write_dataset_to_file(dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


def main():
    for idx, (filename, n_inst, n_items) in enumerate(DATASETS):
        dataset = generate_weibull_dataset(n_inst, n_items, seed=BASE_SEED + idx)
        write_dataset_to_file(dataset, os.path.join(HERE, filename))
        flat = np.concatenate([np.asarray(x) for x in dataset[n_items]])
        print(f'{filename:24} instances={n_inst:>2} items/inst={n_items:>6} '
              f'range=[{flat.min()},{flat.max()}] mean={flat.mean():.2f}')


if __name__ == '__main__':
    main()
