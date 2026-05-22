import numpy as np


class GetData:
    """Random nurse rostering instances.

    Each instance defines:
        n_nurses      – number of nurses available
        n_days        – rostering horizon in days
        n_shift_types – 3: 0=morning, 1=afternoon, 2=night
        requirements  – (n_shift_types,) nurses required per shift per day
        preferences   – (n_nurses, n_shift_types) in {-1, 0, 1}
                        -1=dislike, 0=neutral, 1=prefer
        max_consecutive – maximum allowed consecutive working days (soft)
    """

    @staticmethod
    def generate_instances(n_instances: int, n_nurses: int = 8,
                           n_days: int = 14, seed: int = 42) -> list[dict]:
        rng = np.random.RandomState(seed)
        instances = []
        for _ in range(n_instances):
            # Preferences drawn uniformly from {-1, 0, 1}
            prefs = rng.randint(-1, 2, size=(n_nurses, 3))
            instances.append({
                'n_nurses':       n_nurses,
                'n_days':         n_days,
                'n_shift_types':  3,
                'requirements':   np.array([2, 2, 1], dtype=int),  # morning, afternoon, night
                'preferences':    prefs.astype(float),
                'max_consecutive': 5,
            })
        return instances
