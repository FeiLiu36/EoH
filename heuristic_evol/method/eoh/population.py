from __future__ import annotations

from typing import List

import numpy as np

from ...heuristic import *


class Population:
    def __init__(self):
        self._population: List[Function] = []

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __add__(self, other: Function | List[Function]):
        if isinstance(other, Function):
            self._population.append(other)
        elif isinstance(other, list):
            for f in other:
                if not isinstance(f, Function):
                    raise ValueError('"other" should be instance of "Function" or "List[Function]".')
                self._population.append(f)
        else:
            raise ValueError('"other" should be instance of "Function" or "List[Function]".')

    def append_functions(self, funcs: Function | List[Function]):
        self.__add__(funcs)

    def has_duplicate_function(self, func: str | Function) -> bool:
        for f in self._population:
            if str(f) == str(func):
                return True
        return False

    def select_function_roulette(self) -> Function:
        for p in self._population:
            assert p.score is not None
        score = np.array([p.score for p in self._population])
        score = score / score.sum()
        return np.random.choice(self._population, p=score)

    def select_function_tournament(self, tournament_size=2) -> Function:
        for p in self._population:
            assert p.score is not None
        candidates = np.random.choice(self._population, size=tournament_size)
        return max(candidates, key=lambda p: p.score)

    def get_elite(self) -> Function:
        return max(self._population, key=lambda p: p.score)
