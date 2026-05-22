# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


class NurseRostering(BaseProblem):
    """EoH task: design the shift-assignment priority scoring function for nurse rostering.

    A greedy construction heuristic builds the roster day by day, shift by shift.
    For each (day, shift_type) slot requiring k nurses, the harness scores every
    eligible nurse (not yet assigned that day) using the LLM-designed
    `score_assignment` function and greedily selects the top-k scorers.

    Nurses have preferences for each shift type (in {-1, 0, 1}) and soft constraints
    on consecutive working days and on working a morning shift the day after a night
    shift. The scoring function controls which trade-offs are prioritised.

    Fitness (lower is better):
        workload_std            – std of total shifts per nurse over the horizon
        + (1 − preference_mean) – mean preference of assigned shifts, shifted so
                                  random assignment scores ≈ 1.0
        + 0.2 × consecutive_violations
                                – nurse-days exceeding max_consecutive working days
        + 0.3 × night_morning_violations
                                – morning shifts assigned the day after a night shift
    """

    template_program = '''
def score_assignment(nurse_idx: int, shift_type: int, day: int,
                     nurse_workload: np.ndarray, nurse_preferences: np.ndarray,
                     consecutive_days: np.ndarray, last_shift_type: np.ndarray,
                     target_workload: float, n_days: int) -> float:
    """Score the desirability of assigning a nurse to a particular shift.

    Called once per (eligible nurse, shift_type, day) triple during greedy
    roster construction. The harness assigns the highest-scoring eligible
    nurses to fill each shift slot.

    Args:
        nurse_idx:          index of the nurse being scored (0-based)
        shift_type:         0=morning, 1=afternoon, 2=night
        day:                current day index (0-based)
        nurse_workload:     float array shape (N,) — shifts assigned so far
        nurse_preferences:  float array shape (N, 3) — preference per shift type:
                            -1=dislike, 0=neutral, +1=prefer
        consecutive_days:   int array shape (N,) — consecutive days worked up to
                            yesterday (reset to 0 when a nurse has a day off)
        last_shift_type:    int array shape (N,) — shift type worked on the last
                            working day (-1 if the nurse has not yet been assigned)
        target_workload:    ideal number of shifts per nurse by the end of day `day`
                            assuming perfect balance
        n_days:             total days in the rostering period
    Returns:
        score: higher → more preferred; the harness selects the top-k scorers
               to fill k required slots (ties broken by nurse index)
    """
    preference = nurse_preferences[nurse_idx, shift_type]
    workload_gap = nurse_workload[nurse_idx] - target_workload
    consecutive_penalty = max(0.0, float(consecutive_days[nurse_idx]) - 4.0)
    night_morning_penalty = 1.0 if (shift_type == 0 and last_shift_type[nurse_idx] == 2) else 0.0
    return float(preference - 0.5 * workload_gap - 2.0 * consecutive_penalty
                 - 5.0 * night_morning_penalty)
'''

    task_description = (
        "Design a novel shift-assignment priority scoring function for nurse rostering. "
        "A greedy construction algorithm builds a 2-week roster for 8 nurses across "
        "three daily shift types (morning, afternoon, night) requiring [2, 2, 1] nurses "
        "each. For every shift slot the harness calls your function for each eligible "
        "nurse and assigns the top scorers. "
        "Inputs: the nurse index, the shift type (0=morning, 1=afternoon, 2=night), "
        "the current day, workload assigned so far, nurse preferences (−1/0/+1 per shift "
        "type), consecutive days worked, last shift type worked, and the target workload "
        "for a balanced schedule. "
        "The baseline linearly combines preference, workload deviation, consecutive-day "
        "penalty, and night-to-morning penalty. "
        "Alternatives include: non-linear preference boosting, adaptive weight schedules "
        "that shift from balance-focus early to preference-focus late, lookahead terms "
        "that estimate future constraint risk, or interaction terms between workload "
        "imbalance and preference satisfaction. "
        "Fitness (lower is better): workload standard deviation + (1 − mean preference "
        "of assigned shifts) + weighted soft-constraint violation counts."
    )

    def __init__(self, n_nurses: int = 8, n_days: int = 14, n_instances: int = 5,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_nurses = n_nurses
        self.n_days = n_days
        self.instances = GetData.generate_instances(n_instances, n_nurses, n_days, seed=42)

    def _construct_roster(self, instance: dict, score_fn) -> dict:
        """Greedy day-by-day, shift-by-shift roster construction."""
        n = instance['n_nurses']
        n_days = instance['n_days']
        prefs = instance['preferences']
        reqs = instance['requirements']
        n_shift_types = instance['n_shift_types']

        assignment = np.full((n, n_days), -1, dtype=int)
        workload = np.zeros(n, dtype=float)
        consecutive = np.zeros(n, dtype=int)
        last_shift = np.full(n, -1, dtype=int)

        total_per_day = int(reqs.sum())

        for day in range(n_days):
            target = float((day + 1) * total_per_day) / n
            assigned_today: set[int] = set()

            for shift_type in range(n_shift_types):
                needed = int(reqs[shift_type])
                eligible = [i for i in range(n) if i not in assigned_today]
                if len(eligible) < needed:
                    continue  # coverage failure (shouldn't happen with 8 nurses, 5 slots)

                scored: list[tuple[float, int]] = []
                for ni in eligible:
                    try:
                        s = float(score_fn(
                            int(ni), int(shift_type), int(day),
                            workload.copy(), prefs,
                            consecutive.copy(), last_shift.copy(),
                            float(target), int(n_days),
                        ))
                    except Exception:
                        s = 0.0
                    if not np.isfinite(s):
                        s = 0.0
                    scored.append((s, ni))

                scored.sort(key=lambda x: (-x[0], x[1]))  # desc score, asc index for ties
                for _, ni in scored[:needed]:
                    assignment[ni, day] = shift_type
                    workload[ni] += 1
                    assigned_today.add(ni)
                    last_shift[ni] = shift_type

            # Update consecutive days after this day
            for ni in range(n):
                if ni in assigned_today:
                    consecutive[ni] += 1
                else:
                    consecutive[ni] = 0

        return {'assignment': assignment, 'workload': workload,
                'prefs': prefs, 'max_consecutive': instance['max_consecutive']}

    def _compute_fitness(self, result: dict) -> float:
        assignment = result['assignment']
        workload = result['workload']
        prefs = result['prefs']
        max_consec = result['max_consecutive']
        n, n_days = assignment.shape

        # Workload balance
        workload_std = float(np.std(workload))

        # Preference satisfaction
        pref_vals = [
            prefs[ni, assignment[ni, d]]
            for ni in range(n)
            for d in range(n_days)
            if assignment[ni, d] >= 0
        ]
        preference_mean = float(np.mean(pref_vals)) if pref_vals else 0.0

        # Consecutive-day violations: count nurse-days where consecutive > max_consecutive
        consec_violations = 0
        for ni in range(n):
            run = 0
            for d in range(n_days):
                if assignment[ni, d] >= 0:
                    run += 1
                    if run > max_consec:
                        consec_violations += 1
                else:
                    run = 0

        # Night-to-morning violations
        nm_violations = 0
        for ni in range(n):
            for d in range(1, n_days):
                if assignment[ni, d - 1] == 2 and assignment[ni, d] == 0:
                    nm_violations += 1

        return (workload_std
                + (1.0 - preference_mean)
                + 0.2 * consec_violations
                + 0.3 * nm_violations)

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        scores = []
        for instance in self.instances:
            result = self._construct_roster(instance, callable_func)
            scores.append(self._compute_fitness(result))
        return float(np.mean(scores))
