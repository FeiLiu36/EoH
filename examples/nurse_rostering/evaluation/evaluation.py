import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData


class Evaluation:
    """Post-hoc evaluator for nurse-rostering scoring functions.

    Uses more instances and longer horizons than the training evaluator in prob.py.
    Tests on 14-day (2-week) and 21-day (3-week) rosters with 8 nurses.
    """

    CONFIGS = [
        {'n_nurses': 8, 'n_days': 14, 'n_instances': 10, 'seed': 200},
        {'n_nurses': 8, 'n_days': 21, 'n_instances': 10, 'seed': 300},
    ]

    def __init__(self):
        self.groups = []
        for cfg in self.CONFIGS:
            instances = GetData.generate_instances(
                cfg['n_instances'], cfg['n_nurses'], cfg['n_days'], seed=cfg['seed']
            )
            self.groups.append({
                'label':     f"{cfg['n_nurses']} nurses × {cfg['n_days']} days",
                'n_nurses':  cfg['n_nurses'],
                'n_days':    cfg['n_days'],
                'instances': instances,
            })

    def _construct_roster(self, instance: dict, score_fn) -> dict:
        n = instance['n_nurses']
        n_days = instance['n_days']
        prefs = instance['preferences']
        reqs = instance['requirements']
        n_shift_types = instance['n_shift_types']
        total_per_day = int(reqs.sum())

        assignment = np.full((n, n_days), -1, dtype=int)
        workload = np.zeros(n, dtype=float)
        consecutive = np.zeros(n, dtype=int)
        last_shift = np.full(n, -1, dtype=int)

        for day in range(n_days):
            target = float((day + 1) * total_per_day) / n
            assigned_today: set[int] = set()

            for shift_type in range(n_shift_types):
                needed = int(reqs[shift_type])
                eligible = [i for i in range(n) if i not in assigned_today]
                if len(eligible) < needed:
                    continue

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

                scored.sort(key=lambda x: (-x[0], x[1]))
                for _, ni in scored[:needed]:
                    assignment[ni, day] = shift_type
                    workload[ni] += 1
                    assigned_today.add(ni)
                    last_shift[ni] = shift_type

            for ni in range(n):
                if ni in assigned_today:
                    consecutive[ni] += 1
                else:
                    consecutive[ni] = 0

        return {'assignment': assignment, 'workload': workload,
                'prefs': prefs, 'max_consecutive': instance['max_consecutive']}

    def _compute_metrics(self, result: dict) -> dict:
        assignment = result['assignment']
        workload = result['workload']
        prefs = result['prefs']
        max_consec = result['max_consecutive']
        n, n_days = assignment.shape

        workload_std = float(np.std(workload))

        pref_vals = [
            prefs[ni, assignment[ni, d]]
            for ni in range(n)
            for d in range(n_days)
            if assignment[ni, d] >= 0
        ]
        preference_mean = float(np.mean(pref_vals)) if pref_vals else 0.0

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

        nm_violations = sum(
            1
            for ni in range(n)
            for d in range(1, n_days)
            if assignment[ni, d - 1] == 2 and assignment[ni, d] == 0
        )

        composite = (workload_std
                     + (1.0 - preference_mean)
                     + 0.2 * consec_violations
                     + 0.3 * nm_violations)
        return {
            'workload_std':     workload_std,
            'preference_mean':  preference_mean,
            'consec_violations': consec_violations,
            'nm_violations':    nm_violations,
            'composite':        composite,
        }

    def evaluate(self, score_fn) -> list[dict]:
        """Evaluate score_fn on all benchmark groups.

        Returns list of dicts with keys: label, mean_composite, std_composite,
        mean_workload_std, mean_preference_mean, mean_consec_violations, mean_nm_violations.
        """
        results = []
        for group in self.groups:
            metrics_list = []
            for instance in group['instances']:
                result = self._construct_roster(instance, score_fn)
                metrics_list.append(self._compute_metrics(result))

            results.append({
                'label':                 group['label'],
                'mean_composite':        float(np.mean([m['composite'] for m in metrics_list])),
                'std_composite':         float(np.std([m['composite'] for m in metrics_list])),
                'mean_workload_std':     float(np.mean([m['workload_std'] for m in metrics_list])),
                'mean_preference_mean':  float(np.mean([m['preference_mean'] for m in metrics_list])),
                'mean_consec_violations': float(np.mean([m['consec_violations'] for m in metrics_list])),
                'mean_nm_violations':    float(np.mean([m['nm_violations'] for m in metrics_list])),
            })
        return results
