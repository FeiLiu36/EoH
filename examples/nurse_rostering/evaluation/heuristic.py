# Baseline heuristic: linear combination of preference, workload balance, and soft
# constraint penalties. Replace this body with the best function found by EoH.

import numpy as np


def score_assignment(nurse_idx: int, shift_type: int, day: int,
                     nurse_workload: np.ndarray, nurse_preferences: np.ndarray,
                     consecutive_days: np.ndarray, last_shift_type: np.ndarray,
                     target_workload: float, n_days: int) -> float:
    import numpy as np
    preference = nurse_preferences[nurse_idx, shift_type]
    workload_gap = nurse_workload[nurse_idx] - target_workload
    consecutive_penalty = max(0.0, float(consecutive_days[nurse_idx]) - 4.0)
    night_morning_penalty = 1.0 if (shift_type == 0 and last_shift_type[nurse_idx] == 2) else 0.0
    return float(preference - 0.5 * workload_gap - 2.0 * consecutive_penalty
                 - 5.0 * night_morning_penalty)
