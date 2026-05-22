# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import itertools
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem


# Encoding constants
_TRIPLES = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1), (2, 2, 2)]
_INT_TO_WEIGHT = [0, 1, 1, 2, 2, 3, 3]

# Known optimal admissible-set cardinalities for supported (n, w) pairs
_OPTIMAL = {
    (12, 7):  792,
    (15, 10): 3003,
    (21, 15): 43596,
    (24, 17): 237984,
}

_BAD_TRIPLES = {
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4), (0, 0, 5), (0, 0, 6),
    (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5), (0, 6, 6),
    (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 3), (1, 2, 4),
    (1, 3, 3), (1, 4, 4), (1, 5, 5), (1, 6, 6),
    (2, 2, 2), (2, 3, 3), (2, 4, 4), (2, 5, 5), (2, 6, 6),
    (3, 3, 3), (3, 3, 4), (3, 4, 4), (3, 4, 5), (3, 4, 6), (3, 5, 5), (3, 6, 6),
    (4, 4, 4), (4, 5, 5), (4, 6, 6),
    (5, 5, 5), (5, 5, 6), (5, 6, 6),
    (6, 6, 6),
}


class AdmissibleSet(BaseProblem):
    """Symmetric constant-weight admissible set problem.

    The LLM designs a priority function that scores candidate elements for
    inclusion in a symmetric constant-weight admissible set I(n, w).
    A greedy search uses these scores to build the set incrementally.

    Fitness: optimal_size - achieved_size  (lower is better; 0 = matches known
             optimal; negative = exceeds known optimal).
    """

    template_program = '''
def priority(el: tuple, n: int, w: int) -> float:
    """Score a candidate element for inclusion in the admissible set.

    Args:
        el: candidate element — a tuple of length n whose non-zero entries
            sum to w (entries are drawn from {0, 1, 2}).
        n:  vector length (dimension of the problem).
        w:  target weight (number of non-zero entries).
    Returns:
        Priority score. Higher scores are selected first by the greedy search.
    """
    return 0.0
'''

    task_description = (
        "Design a novel priority function for a greedy algorithm that constructs "
        "a maximum-cardinality symmetric constant-weight admissible set I(n, w). "
        "At each step the greedy search scores all remaining candidate elements "
        "using your priority function and picks the highest-scoring one. "
        "The goal is to maximise the size of the resulting admissible set."
    )

    def __init__(self, dimension: int = 15, weight: int = 10,
                 timeout: int = 60, n_processes: int = 1):
        if (dimension, weight) not in _OPTIMAL:
            raise ValueError(
                f"Unsupported (dimension, weight)=({dimension},{weight}). "
                f"Supported pairs: {list(_OPTIMAL.keys())}"
            )
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.dimension = dimension
        self.weight = weight
        self.num_groups = dimension // 3
        self.optimal_size = _OPTIMAL[(dimension, weight)]

        # Pre-compute all valid children (done once at init, not per evaluation)
        self._valid_children = [
            np.array(child, dtype=np.int32)
            for child in itertools.product(range(7), repeat=self.num_groups)
            if sum(_INT_TO_WEIGHT[x] for x in child) == weight
        ]

    # ── evaluation helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _get_surviving_indices(extant, new_el, candidates):
        surviving = []
        for idx, child in enumerate(candidates):
            # Skip if new_el dominates child or vice versa
            if all(_INT_TO_WEIGHT[x] <= _INT_TO_WEIGHT[y]
                   for x, y in zip(new_el, child)):
                continue
            if all(_INT_TO_WEIGHT[x] >= _INT_TO_WEIGHT[y]
                   for x, y in zip(new_el, child)):
                continue
            # Check bad-triple constraint against all extant elements
            invalid = False
            for ext in extant:
                if all(tuple(sorted((int(x), int(y), int(z)))) in _BAD_TRIPLES
                       for x, y, z in zip(ext, new_el, child)):
                    invalid = True
                    break
            if not invalid:
                surviving.append(idx)
        return surviving

    @staticmethod
    def _expand(pre_admissible_set, num_groups):
        result = []
        for row in pre_admissible_set:
            rotations = [[] for _ in range(num_groups)]
            for i in range(num_groups):
                x, y, z = _TRIPLES[row[i]]
                rotations[i].append((x, y, z))
                if not (x == y == z):
                    rotations[i].append((z, x, y))
                    rotations[i].append((y, z, x))
            for combo in itertools.product(*rotations):
                result.append(sum(combo, ()))
        return result

    # ── EoH interface ──────────────────────────────────────────────────────────

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        candidates = list(self._valid_children)     # mutable copy per evaluation
        scores = np.array([
            callable_func(
                sum((_TRIPLES[x] for x in el), ()),
                self.dimension,
                self.weight,
            )
            for el in candidates
        ], dtype=float)

        pre_admissible = np.empty((0, self.num_groups), dtype=np.int32)

        while candidates:
            best = int(np.argmax(scores))
            chosen = candidates[best]
            surviving = self._get_surviving_indices(pre_admissible, chosen, candidates)
            candidates = [candidates[i] for i in surviving]
            scores = scores[surviving]
            pre_admissible = np.concatenate([pre_admissible, chosen[None]], axis=0)

        admissible_set = self._expand(pre_admissible, self.num_groups)
        return float(self.optimal_size - len(admissible_set))
